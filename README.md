# MRPT notebooks

Interactive [marimo](https://marimo.io) notebooks showing the
[MRPT](https://www.mrpt.org/) Python API, installed from PyPI with
`pip install mrpt3`. Each one explains a robotics concept and lets you change
its parameters with sliders.

The molab links open a static preview with the saved outputs. To move the
sliders, run the notebook on a server from the preview menu (or locally, see
below).

| Topic | Notebook | MRPT modules |
|---|---|---|
| **Poses** | [3D poses in SE(3)](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/se3_poses.py) | `mrpt.poses` |
| | [Odometry uncertainty: Gaussian vs. particles](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/pose_uncertainty.py) | `mrpt.poses`, `mrpt.obs`, `mrpt.slam`, `mrpt.bayes` |
| | [Rigid transformations from point correspondences](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/rigid_transform_estimation.py) | `mrpt.tfest`, `mrpt.poses` |
| **Registration and SLAM** | [2D scan matching with ICP](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/icp_scan_matching.py) | `mrpt.slam`, `mrpt.maps`, `mrpt.poses` |
| | [Pose graphs and loop closures](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/pose_graph_slam.py) | `mrpt.graphs`, `mrpt.poses` |
| | [Monte Carlo localization with a particle filter](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/monte_carlo_localization.py) | `mrpt.slam`, `mrpt.bayes`, `mrpt.maps`, `mrpt.obs` |
| **Maps** | [Occupancy grid mapping from laser scans](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/occupancy_grid_mapping.py) | `mrpt.maps`, `mrpt.obs` |
| | [3D metric maps from a point cloud](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/point_cloud_maps.py) | `mrpt.maps`, `mrpt.obs` |
| **Robot arms** | [Denavit-Hartenberg parameters](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/dh_parameters.py) | `mrpt.poses` |
| | [Forward kinematics, workspace and Jacobian](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/forward_kinematics.py) | `mrpt.poses` |
| | [Inverse kinematics: closed form and numerical](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/inverse_kinematics.py) | `mrpt.poses` |
| **Navigation and kinematics** | [Trajectory families and TP-Space (PTGs)](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/ptg_trajectories.py) | `mrpt.nav` |
| | [Mobile robot kinematics: differential drive vs. holonomic](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/vehicle_kinematics.py) | `mrpt.kinematics`, `mrpt.obs` |
| **Sensors and geodesy** | [Camera models and lens distortion](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/camera_models.py) | `mrpt.img`, `mrpt.obs` |
| | [Geodetic coordinates: WGS84, ECEF, ENU and UTM](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/geodetic_coordinates.py) | `mrpt.topography` |
| **Visualization and tools** | [Off-screen 3D rendering](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/offscreen_3d_rendering.py) | `mrpt.viz`, `mrpt.opengl` |
| | [Math formulas compiled at runtime](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/runtime_expressions.py) | `mrpt.expr`, `mrpt.config` |

## Run locally

Requires Linux (x86_64 or aarch64) and Python 3.11 to 3.14, the platforms of
the `mrpt3` wheels. With [uv](https://docs.astral.sh/uv/):

```bash
uvx marimo edit --sandbox notebooks/icp_scan_matching.py
```

`--sandbox` installs the dependencies listed at the top of each notebook
(`mrpt3`, `numpy`, `matplotlib`...) in a temporary environment. The PyPI
package is `mrpt3`, although the import name is `mrpt`: if a tool offers to
install a package called `mrpt`, that is an unrelated project.

The off-screen rendering notebook also needs the system OpenGL/EGL libraries
(on Debian/Ubuntu: `libegl1 libgl1 libegl-mesa0 libgl1-mesa-dri`); without
them it shows how to install them.

## Saved outputs

The previews show the outputs stored in `notebooks/__marimo__/session/`.
After changing a notebook, refresh them with:

```bash
uvx marimo export session --sandbox notebooks/
```
