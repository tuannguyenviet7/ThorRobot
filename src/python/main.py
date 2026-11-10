"""
Trajectory planning for a MuJoCo robot (works with any model without free joints).

Features
- Minimum-jerk (quintic) joint-space trajectories through several waypoints
- Optional Cartesian goal -> joint goal via damped-least-squares inverse kinematics
- Two ways to follow the plan:
    "position": send q_des to position actuators
    "torque":   PD + gravity compensation for motor (torque) actuators

Assumptions
- Each actuator drives one joint (hinge/slide). Mapping is read from the model.
- Model has no free/ball joints (nq == nv).

Notation used in the comments
    q, qd      joint positions / velocities           (n-vector)
    q_des      desired joint positions (from planner)
    qd_des     desired joint velocities (from planner)
    x          end-effector position in world frame   (3-vector)
    J(q)       positional Jacobian, J = dx/dq         (3 x n matrix)
    T          duration of one trajectory segment [s]
"""
import time

import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "/home/ngvietuan/ThorRobot/ThorRobot/sim/main.xml"

CONTROL_MODE = "position"   # "position" or "torque"

# PD gains for torque mode. Per joint: tau = KP*(position error) + KD*(velocity error).
# Rule of thumb for a joint with effective inertia I:
#   natural frequency  w = sqrt(KP / I)
#   critical damping   KD = 2 * sqrt(KP * I)
# Increase KP for tighter tracking, then raise KD until overshoot disappears.
KP, KD = 100.0, 10.0

SEGMENT_TIME = 3.0          # T: seconds per waypoint-to-waypoint segment
HOLD_TIME = 2.0             # seconds to hold the final pose before closing
EE_SITE = None              # e.g. "ee_site" to enable the Cartesian example below


# ==========================================================================
# 1) Trajectory generation
# ==========================================================================
def min_jerk(tau):
    """
    Minimum-jerk time scaling.

    We want a smooth scalar profile s(tau), tau in [0, 1], going from 0 to 1
    with zero velocity AND zero acceleration at both ends (so the robot starts
    and stops gently, with no jerk spikes). That gives 6 boundary conditions:

        s(0)=0,  s'(0)=0,  s''(0)=0
        s(1)=1,  s'(1)=0,  s''(1)=0

    A 5th-order polynomial has 6 coefficients, so it fits exactly:

        s(tau)  = 10*tau^3 - 15*tau^4 + 6*tau^5          (position scaling)
        s'(tau) = 30*tau^2 - 60*tau^3 + 30*tau^4         (velocity scaling, d s / d tau)

    (This polynomial is also the one that minimises the integral of squared jerk.)
    """
    tau = np.clip(tau, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    return s, ds


class JointTrajectory:
    """Piecewise min-jerk trajectory through waypoints (rest-to-rest at each one)."""

    def __init__(self, waypoints, segment_time):
        self.wp = [np.asarray(w, dtype=float) for w in waypoints]
        self.T = segment_time
        self.n_seg = len(self.wp) - 1          # N waypoints -> N-1 segments
        self.duration = self.n_seg * self.T    # total motion time

    def sample(self, t):
        """
        Return (q_des, qd_des) at time t.

        Inside segment i (waypoints wp[i] -> wp[i+1]):
            tau    = (t - i*T) / T                 normalised time in [0, 1]
            dq     = wp[i+1] - wp[i]               total joint displacement of the segment
            q(t)   = wp[i] + s(tau) * dq           interpolated position
            qd(t)  = s'(tau) * dq / T              chain rule: d/dt = (1/T) * d/dtau
        """
        # After the last segment: stay at the final waypoint with zero velocity.
        if t >= self.duration:
            return self.wp[-1].copy(), np.zeros_like(self.wp[-1])

        i = int(t // self.T)                   # index of the active segment
        tau = (t - i * self.T) / self.T        # normalised time inside the segment
        s, ds = min_jerk(tau)
        dq = self.wp[i + 1] - self.wp[i]
        return self.wp[i] + s * dq, ds * dq / self.T


# ==========================================================================
# 2) Inverse kinematics (position only, damped least squares)
# ==========================================================================
def solve_ik(m, site_name, target_pos, q_init, iters=300, tol=1e-4,
             damping=1e-2, step=0.5):
    """
    Find joint angles q such that the site position x(q) reaches target_pos.

    Forward kinematics is nonlinear, but for a small joint change dq it is
    approximately linear:

        x(q + dq) ~= x(q) + J(q) * dq

    We want x(q + dq) = target, i.e.  J * dq = e,  with error  e = target - x(q).
    Instead of the plain pseudo-inverse (unstable near singularities), we use
    damped least squares, which minimises

        || J*dq - e ||^2  +  lambda^2 * || dq ||^2

    and has the closed-form solution

        dq = J^T * (J*J^T + lambda^2 * I)^-1 * e

    - lambda (=damping) trades accuracy for stability: large values keep dq
      small near singular poses where J loses rank.
    - J*J^T is only 3x3 (position task), so the solve is cheap.
    - `step` scales dq (0 < step <= 1) to avoid overshooting the linearisation.
    The update is repeated until ||e|| < tol (Newton-like iteration).
    """
    # Work on a private copy of the data so the live simulation is not disturbed.
    d = mujoco.MjData(m)
    d.qpos[:] = q_init
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site '{site_name}' not found in model")
    jacp = np.zeros((3, m.nv))                 # positional Jacobian J (3 x nv)

    for _ in range(iters):
        mujoco.mj_fwdPosition(m, d)            # compute x(q) and prepare Jacobian data
        err = np.asarray(target_pos) - d.site_xpos[site_id]   # e = target - x(q)
        if np.linalg.norm(err) < tol:          # converged
            break
        mujoco.mj_jacSite(m, d, jacp, None, site_id)          # fill J(q)

        # Damped least-squares step: dq = J^T (J J^T + lambda^2 I)^-1 e
        dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + damping**2 * np.eye(3), err)

        # q <- q + step*dq  (mj_integratePos is the proper way to add to qpos)
        mujoco.mj_integratePos(m, d.qpos, dq, step)

        # Clamp every limited hinge/slide joint into its allowed range.
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] in (
                mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
            ):
                a = m.jnt_qposadr[j]
                d.qpos[a] = np.clip(d.qpos[a], *m.jnt_range[j])
    else:
        # for-loop finished without `break`: never reached the tolerance
        print("IK: max iterations reached, error =", np.linalg.norm(err))
    return d.qpos.copy()


# ==========================================================================
# 3) Controller: map joint-space reference to actuator commands
# ==========================================================================
def apply_control(m, d, q_des, qd_des, mode):
    """
    position mode:  ctrl = q_des
        MuJoCo's <position> actuator already contains a PD law internally:
            force = kp * (ctrl - q) - kv * qd

    torque mode:    ctrl = tau, computed here as a PD law with feed-forward
        tau = KP*(q_des - q) + KD*(qd_des - qd) + qfrc_bias

        - KP*(q_des - q)     proportional term: pulls q towards q_des
        - KD*(qd_des - qd)   derivative term: damps motion and tracks velocity
        - qfrc_bias          = C(q,qd)*qd + g(q), Coriolis/centrifugal + gravity.
                             Adding it cancels those terms, because the robot
                             dynamics are   M(q)*qdd + qfrc_bias = tau.
                             With the bias cancelled, M*qdd = PD, so the error
                             behaves like a damped spring.
    """
    for a in range(m.nu):
        # Which joint does actuator `a` drive?  (trnid[:,0] holds the joint id)
        jid = m.actuator_trnid[a, 0]
        qadr = m.jnt_qposadr[jid]              # index of this joint in qpos
        dadr = m.jnt_dofadr[jid]               # index of this joint in qvel / qfrc_bias

        if mode == "position":
            u = q_des[qadr]
        else:
            u = (KP * (q_des[qadr] - d.qpos[qadr])
                 + KD * (qd_des[dadr] - d.qvel[dadr])
                 + d.qfrc_bias[dadr])

        # Saturate to the actuator's command range if one is defined.
        if m.actuator_ctrllimited[a]:
            u = np.clip(u, *m.actuator_ctrlrange[a])
        d.ctrl[a] = u


# ==========================================================================
# 4) Main
# ==========================================================================
def main():
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    assert m.nq == m.nv, "This script assumes no free/ball joints."

    q_start = d.qpos.copy()

    # ---- Define waypoints (joint space) ---------------------------------
    # Example: move the first 3 joints by +0.4 rad, then -0.4 rad, then return.
    # Each segment takes SEGMENT_TIME seconds, so the full motion lasts
    #   duration = (number_of_waypoints - 1) * SEGMENT_TIME = 3 * 3.0 = 9 s here.
    # Peak velocity of a min-jerk segment is 1.875 * |dq| / T (at tau = 0.5),
    # so check it against your joint velocity limits: 1.875 * 0.4 / 3.0 = 0.25 rad/s.
    q_a = q_start.copy()
    q_a[:3] += 0.4
    q_b = q_start.copy()
    q_b[:3] -= 0.4
    waypoints = [q_start, q_a, q_b, q_start]

    # ---- Or: plan to a Cartesian target through IK ----------------------
    if EE_SITE is not None:
        mujoco.mj_forward(m, d)                # compute site positions at q_start
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        # Target = current end-effector position + an offset in world coordinates [m].
        target = d.site_xpos[sid] + np.array([0.0, 0.1, -0.05])  # 10 cm +y, 5 cm down
        q_goal = solve_ik(m, EE_SITE, target, q_start)
        waypoints = [q_start, q_goal, q_start]  # go there and come back

    traj = JointTrajectory(waypoints, SEGMENT_TIME)
    t_end = traj.duration + HOLD_TIME

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and d.time < t_end:
            step_start = time.time()

            # Planner uses *simulation* time d.time, so the motion is the same
            # no matter how fast the loop runs.
            q_des, qd_des = traj.sample(d.time)
            apply_control(m, d, q_des, qd_des, CONTROL_MODE)
            mujoco.mj_step(m, d)               # advance physics by m.opt.timestep
            viewer.sync()

            # Real-time pacing: sleep for the remainder of one physics timestep.
            dt = m.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()