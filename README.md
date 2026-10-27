# ThorRobot

MuJoCo simulation of a 6-DOF robotic arm, built on the open-source [Thor](https://github.com/AngelLM/Thor) design by Ángel LM (`.stl` files).

Thor is a 3D-printed arm with a yaw-roll-roll-yaw-roll-yaw joint layout, about 625 mm tall. This project turns its printable parts into a simulation model you can load, move and control.

## Features

- MuJoCo model (`sim/main.xml`) with a fixed base, six hinge joints and a rigid gripper
- Part placement taken from the original Thor FreeCAD assemblies, so the home pose matches the real arm
- Meshes organised into three groups: `body`, `arm` and `gripper`
- Position actuators for all six joints and a `home` keyframe

## Project structure
"IN PROGRESS"
```
sim/
├── main.xml                  # MuJoCo model
└── meshes/
    └── collision/
        ├── arm/              # links 1-6 (Art* parts, bearing holders)
        ├── body/             # fixed base and electronics box
        └── gripper/          # Gripper* parts
```

## Getting started

Requires Python 3.9 or newer.

```bash
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

# install dependencies
pip install mujoco numpy

# open the interactive viewer
python -m mujoco.viewer --mjcf=sim/main.xml
```

On macOS, start the viewer with `mjpython` instead of `python`.

Use the control sliders in the viewer to move each joint. Joint angles are in radians, and 0 is the home pose (arm straight up).

### Loading the model from Python

```python
import mujoco, mujoco.viewer

model = mujoco.MjModel.from_xml_path("sim/main.xml")
data = mujoco.MjData(model)
data.ctrl[:] = [0.5, 0.3, -0.4, 0, 0.2, 0]   # target angles for joints 1-6

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

## The model

| Joint | Motion | Role |
|-------|--------|------|
| j1 | yaw | base rotation |
| j2 | roll | shoulder |
| j3 | roll | elbow |
| j4 | yaw | forearm rotation |
| j5 | roll | wrist bend |
| j6 | yaw | wrist rotation |

Units are metres, kilograms and radians.

## Current limitations

- Meshes are used for appearance only and collisions are switched off. Adding contact needs simplified or convex-decomposed collision shapes.
- The gripper is rigid, with no finger joint.
- The real wrist is a differential drive. Here joints 5 and 6 are independent hinges.
- Only printed parts have mass (about 1.2 kg, assuming roughly PLA at partial infill). Motors, bearings and fans are not modelled.
- Joint ranges are ±180° and the actuator gains are rough defaults. Set them to your hardware's real limits.
- Three Thor parts are not placed in the original assemblies and are left out of the model: `Art1OptoFix`, `Art4BearingPlug` and `Art56Interface`.

## Credits and license

The mechanical design and STL files come from the [Thor project](https://github.com/AngelLM/Thor) by Ángel LM, released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Because the meshes are derived from that work, this project's model and meshes are shared under the same license, with attribution to the original author.