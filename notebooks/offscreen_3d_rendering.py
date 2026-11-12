# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "mrpt3",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Off-screen 3D rendering

    MRPT describes 3D scenes with a scene graph (`mrpt.viz`): a `Scene` holds
    objects such as grids, boxes, point clouds, robot models or uncertainty
    ellipsoids. `mrpt.opengl.CFBORender` renders that scene into an image
    **without any window or display**, using EGL and an OpenGL framebuffer
    object (FBO). It returns the color image and the depth of every pixel, as
    NumPy arrays: useful to make figures, videos, or synthetic camera data on
    a headless server.

    The scene below shows a robot that has driven along a path, with the 95%
    confidence ellipsoids of its position growing as odometry errors
    accumulate. **Move the sliders** to orbit the camera: each frame is
    rendered on the server.
    """)
    return


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mrpt.math import CMatrixDouble33, TPoint3D
    from mrpt.poses import CPose3D
    from mrpt.viz import (
        CBox,
        CEllipsoid3D,
        CGridPlaneXY,
        CPointCloudColoured,
        CSetOfLines,
        CSphere,
        Scene,
        stock_objects,
    )
    return (
        CBox,
        CEllipsoid3D,
        CGridPlaneXY,
        CMatrixDouble33,
        CPointCloudColoured,
        CPose3D,
        CSetOfLines,
        CSphere,
        Scene,
        TPoint3D,
        mo,
        np,
        plt,
        stock_objects,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    # Off-screen rendering needs the system EGL and OpenGL libraries:
    try:
        from mrpt.opengl import CFBORender

        fbo_error = None
    except ImportError as _e:
        CFBORender = None
        fbo_error = str(_e)

    mo.stop(
        fbo_error is not None,
        mo.callout(
            mo.md(
                f"""
    **`mrpt.opengl` is not available here**: `{fbo_error}`.

    Install the system OpenGL/EGL libraries and restart the notebook. On Debian/Ubuntu:

    ```bash
    sudo apt install libegl1 libegl-mesa0 libgl1-mesa-dri libgl1 libgles2 libopengl0 \
      libx11-6 libx11-xcb1 libxcursor1 libxext6 libxi6 libxinerama1 libxrandr2 libxrender1 libxxf86vm1
    ```

    Without a GPU, Mesa renders on the CPU (llvmpipe), which is fast enough for this scene.
    """
            ),
            kind="warn",
        ),
    )
    return (CFBORender,)


@app.cell(hide_code=True)
def _(mo):
    azimuth = mo.ui.slider(start=-180, stop=180, step=5, value=-50, label="Azimuth (deg)", show_value=True)
    elevation = mo.ui.slider(start=5, stop=89, step=1, value=30, label="Elevation (deg)", show_value=True)
    zoom = mo.ui.slider(start=5.0, stop=40.0, step=0.5, value=18.0, label="Distance (m)", show_value=True)
    show_cloud = mo.ui.checkbox(value=True, label="Scanned point cloud")
    show_ellipsoids = mo.ui.checkbox(value=True, label="Uncertainty ellipsoids")
    mo.vstack(
        [
            mo.md("### Camera and contents"),
            mo.hstack([azimuth, elevation, zoom], justify="start", gap=2),
            mo.hstack([show_cloud, show_ellipsoids], justify="start", gap=2),
        ]
    )
    return azimuth, elevation, show_cloud, show_ellipsoids, zoom


@app.cell
def _(
    CBox,
    CEllipsoid3D,
    CGridPlaneXY,
    CMatrixDouble33,
    CPointCloudColoured,
    CPose3D,
    CSetOfLines,
    CSphere,
    Scene,
    TPoint3D,
    np,
    show_cloud,
    show_ellipsoids,
    stock_objects,
):
    scene = Scene()
    scene.insert(CGridPlaneXY(-8, 8, -8, 8, 0, 1))
    scene.insert(stock_objects.CornerXYZ(1.0))

    # Some furniture:
    for _x0, _y0, _x1, _y1, _h, _rgb in [
        (-8.0, 7.6, 8.0, 8.0, 1.2, (0.75, 0.75, 0.8)),
        (-8.0, -8.0, -7.6, 8.0, 1.2, (0.75, 0.75, 0.8)),
        (2.0, 2.0, 3.5, 3.0, 0.9, (0.85, 0.45, 0.25)),
        (-4.0, -5.0, -2.5, -3.5, 1.6, (0.3, 0.6, 0.35)),
    ]:
        _box = CBox(TPoint3D(_x0, _y0, 0), TPoint3D(_x1, _y1, _h), False)
        _box.setColor(*_rgb)
        scene.insert(_box)
    _ball = CSphere(0.7)
    _ball.setLocation(4.5, -3.0, 0.7)
    _ball.setColor(0.25, 0.45, 0.9)
    scene.insert(_ball)

    # The robot path, a smooth loop, and the robot at its last pose:
    _t = np.linspace(0.0, 1.6 * np.pi, 120)
    path_xy = np.column_stack([5.0 * np.cos(_t) - 1.0, 3.5 * np.sin(_t)])
    _path = CSetOfLines()
    for (_xa, _ya), (_xb, _yb) in zip(path_xy[:-1], path_xy[1:]):
        _path.appendLine(_xa, _ya, 0.05, _xb, _yb, 0.05)
    _path.setColor(1.0, 0.85, 0.0)
    scene.insert(_path)
    _heading = np.arctan2(*(path_xy[-1] - path_xy[-2])[::-1])
    _robot = stock_objects.RobotPioneer()
    _robot.setPose(CPose3D(path_xy[-1, 0], path_xy[-1, 1], 0.0, _heading, 0.0, 0.0))
    _robot.setScale(2.0)
    scene.insert(_robot)

    # 95% confidence ellipsoids of the position, growing with the distance traveled:
    if show_ellipsoids.value:
        for _i in range(10, len(path_xy), 12):
            _s = 0.03 * np.sqrt(_i)
            _cov = CMatrixDouble33(np.diag([_s**2, (0.6 * _s) ** 2, (0.4 * _s) ** 2]))
            _ell = CEllipsoid3D()
            _ell.setCovMatrix(_cov)
            _ell.setQuantiles(2.0)
            _ell.setColor(0.9, 0.2, 0.2, 0.6)
            _ell.setLocation(path_xy[_i, 0], path_xy[_i, 1], 0.3)
            scene.insert(_ell)

    # A point cloud, as if scanned by a 3D sensor: a wavy wall colored by height.
    if show_cloud.value:
        _rng = np.random.default_rng(0)
        _cloud = CPointCloudColoured()
        for _ in range(6000):
            _x = _rng.uniform(-6.0, 6.0)
            _z = _rng.uniform(0.0, 2.5)
            _y = -7.0 + 0.3 * np.sin(1.5 * _x) + _rng.normal(0.0, 0.02)
            _c = _z / 2.5
            _cloud.push_back(_x, _y, _z, _c, 0.3, 1.0 - _c)
        _cloud.setPointSize(3.0)
        scene.insert(_cloud)
    return (scene,)


@app.cell
def _(CFBORender):
    # One renderer (EGL context + framebuffer) reused for all frames:
    fbo = CFBORender(960, 600)
    return (fbo,)


@app.cell
def _(azimuth, elevation, fbo, np, scene, time, zoom):
    _cam = scene.getViewport("main").getCamera()
    _cam.setAzimuthDegrees(azimuth.value)
    _cam.setElevationDegrees(elevation.value)
    _cam.setZoomDistance(zoom.value)

    _t0 = time.perf_counter()
    _img, depth = fbo.render_RGBD(scene)
    render_ms = 1e3 * (time.perf_counter() - _t0)
    rgb = np.array(_img.as_numpy())  # copy: the view is reused by the next render
    return depth, render_ms, rgb


@app.cell(hide_code=True)
def _(depth, mo, np, plt, render_ms, rgb):
    # Depth as a color image (pixels without any object have depth 0):
    _valid = depth > 0
    _d = np.where(_valid, depth, np.nan)
    _lo, _hi = np.nanpercentile(_d, [1, 99]) if _valid.any() else (0.0, 1.0)
    _norm = np.clip((_d - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0)
    depth_rgb = (255 * plt.cm.turbo(np.nan_to_num(_norm))[..., :3]).astype(np.uint8)
    depth_rgb[~_valid] = 0

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([mo.md("**Color** (`render_RGBD()`, RGB)"), mo.image(rgb)]),
                    mo.vstack([mo.md(f"**Depth**: {_lo:.1f} m (blue) to {_hi:.1f} m (red)"), mo.image(depth_rgb)]),
                ],
                widths="equal",
            ),
            mo.md(f"Rendered {rgb.shape[1]}x{rgb.shape[0]} pixels in **{render_ms:.0f} ms**."),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same scene can be shown in a desktop window with
    `mrpt.gui.CDisplayWindowGUI`, or saved to a file and opened later with
    MRPT's `SceneViewer3D` application.

    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14, with the system OpenGL/EGL libraries):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/offscreen_3d_rendering.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
