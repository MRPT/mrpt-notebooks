# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "mrpt3",
#     "numpy",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3D poses in SE(3)

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to work with rigid-body poses in 3D: a translation
    plus a rotation. It shows:

    - How one rotation is written as yaw, pitch and roll angles, as a
      rotation matrix, and as a unit quaternion (`CPose3D`, `CPose3DQuat`,
      `CQuaternionDouble`).
    - Pose composition $\oplus$, inverse composition $\ominus$ and the
      inverse, to chain coordinate frames (robot, camera, observed point).
    - Smooth interpolation between keyframe poses (`CPose3DInterpolator`).

    All of it lives in the `mrpt.poses` and `mrpt.math` modules. **Move the
    sliders** to change the robot pose; the 3D views (drag to rotate them)
    and all numbers update. Try a pitch of ±90° to meet gimbal lock.
    """)
    return


@app.cell
def _():
    from datetime import timedelta

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from mrpt.poses import CPose3D, CPose3DInterpolator, CPose3DQuat
    return CPose3D, CPose3DInterpolator, CPose3DQuat, go, mo, np, timedelta


@app.cell(hide_code=True)
def _(mo):
    x = mo.ui.slider(start=-3, stop=3, step=0.1, value=1.5, label="x (m)", show_value=True)
    y = mo.ui.slider(start=-3, stop=3, step=0.1, value=1.0, label="y (m)", show_value=True)
    z = mo.ui.slider(start=-3, stop=3, step=0.1, value=0.5, label="z (m)", show_value=True)
    yaw = mo.ui.slider(start=-180, stop=180, step=1, value=40, label="yaw (deg)", show_value=True)
    pitch = mo.ui.slider(start=-90, stop=90, step=1, value=-15, label="pitch (deg)", show_value=True)
    roll = mo.ui.slider(start=-180, stop=180, step=1, value=10, label="roll (deg)", show_value=True)

    mo.vstack(
        [
            mo.md("### Robot pose (in the world frame)"),
            mo.hstack([x, y, z], justify="start", gap=2),
            mo.hstack([yaw, pitch, roll], justify="start", gap=2),
        ]
    )
    return pitch, roll, x, y, yaw, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One rotation, three representations

    MRPT uses the yaw-pitch-roll (Z-Y-X) convention: the rotation matrix is
    $R = R_z(\text{yaw})\, R_y(\text{pitch})\, R_x(\text{roll})$. Its columns
    are the robot's own x, y, z axes, expressed in the world frame. A unit
    quaternion $q = (q_r, q_x, q_y, q_z)$ encodes the same rotation with four
    numbers and no singularities ($q$ and $-q$ are the same rotation). The
    homogeneous matrix stacks $R$ and the
    translation $t$, so that a point maps as $p_{world} = R\, p_{robot} + t$.
    """)
    return


@app.cell
def _(CPose3D, CPose3DQuat, np, pitch, roll, x, y, yaw, z):
    robot = CPose3D.FromXYZYawPitchRoll(
        x.value, y.value, z.value, np.radians(yaw.value), np.radians(pitch.value), np.radians(roll.value)
    )
    R = np.asarray(robot.getRotationMatrix())
    H = np.asarray(robot.getHomogeneousMatrix())
    quat = CPose3DQuat(robot).quat  # a CQuaternionDouble (qr, qx, qy, qz)

    # Back from the rotation matrix to yaw, pitch, roll:
    from_R = CPose3D()
    from_R.setRotationMatrix(robot.getRotationMatrix())
    ypr_from_R = np.degrees(from_R.getYawPitchRoll())
    return H, R, quat, robot, ypr_from_R


@app.cell(hide_code=True)
def _(H, R, mo, np, pitch, quat, roll, yaw, ypr_from_R):
    def bmatrix(m):
        rows = [" & ".join(f"{v:+.3f}" for v in row) for row in np.round(m, 3) + 0.0]  # + 0.0: no "-0.000"
        return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"

    _q = np.round(quat.as_numpy(), 4) + 0.0
    _lock = abs(pitch.value) == 90
    _recovered = mo.md(
        f"Yaw, pitch, roll recovered from $R$: **{ypr_from_R[0]:.1f}°, {ypr_from_R[1]:.1f}°, "
        f"{ypr_from_R[2]:.1f}°** (sliders: {yaw.value}°, {pitch.value}°, {roll.value}°)."
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md(f"**Rotation matrix** `getRotationMatrix()`\n\n$$R = {bmatrix(R)}$$"),
                    mo.md(f"**Homogeneous matrix** `getHomogeneousMatrix()`\n\n$$T = {bmatrix(H)}$$"),
                ],
                justify="start",
                gap=3,
            ),
            mo.md(
                f"**Quaternion** `CPose3DQuat(robot).quat`: $q = ({_q[0]:+.4f},\\ {_q[1]:+.4f},\\ "
                f"{_q[2]:+.4f},\\ {_q[3]:+.4f})$, with $|q|^2 = {quat.normSqr():.6f}$. "
                f"Its rotation matrix, `quat.rotationMatrix()`, differs from $R$ by "
                f"{np.abs(np.asarray(quat.rotationMatrix()) - R).max():.1e}."
            ),
            mo.callout(
                mo.vstack(
                    [
                        _recovered,
                        mo.md(
                            "**Gimbal lock**: at pitch ±90° yaw and roll rotate about the same axis, "
                            "so only their difference (or sum) is defined. Other angles give the same $R$."
                        ),
                    ]
                ),
                kind="warn",
            )
            if _lock
            else _recovered,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Chaining frames: composition and inverse

    A camera is mounted on the robot, 0.5 m ahead and 0.8 m up, tilted 20°
    down. That mounting pose $B$ is given in the robot frame, so the camera
    pose in the world is the composition $A \oplus B$ of the robot pose $A$
    and $B$. The camera sees a point 2.5 m ahead, along its own x axis.

    The inverse composition undoes it: $(A \oplus B) \ominus A = B$, where
    $C \ominus A = A^{-1} \oplus C$ is "the pose of $C$ as seen from $A$". With
    the identity pose $I$, $I \ominus A$ is just the inverse $A^{-1}$.
    """)
    return


@app.cell
def _(CPose3D, np, robot):
    camera_on_robot = CPose3D(0.5, 0.0, 0.8, 0.0, np.radians(20), 0.0)  # B, in the robot frame
    camera = robot + camera_on_robot  # A (+) B: the camera in the world frame
    point_world = camera.composePoint(2.5, 0.0, 0.0)  # a point in the camera frame, to world
    point_robot = robot.inverseComposePoint(point_world)  # the same point, in the robot frame

    recovered_mount = camera - robot  # (A (+) B) (-) A, should equal B
    robot_inv = CPose3D() - robot  # the inverse pose (robot.inverse() would invert it in place)
    identity = robot + robot_inv
    return (
        camera,
        camera_on_robot,
        identity,
        point_robot,
        point_world,
        recovered_mount,
        robot_inv,
    )


@app.cell(hide_code=True)
def _(
    camera,
    camera_on_robot,
    identity,
    mo,
    np,
    point_robot,
    point_world,
    recovered_mount,
    robot,
    robot_inv,
):
    def row(name, p):
        xyz = np.round([p.x, p.y, p.z], 3) + 0.0  # + 0.0: no "-0.000"
        ypr = np.round(np.degrees([p.yaw, p.pitch, p.roll]), 1) + 0.0
        return f"| {name} | " + " | ".join([f"{v:.3f}" for v in xyz] + [f"{v:.1f}" for v in ypr]) + " |"

    mo.md(
        "\n".join(
            [
                "| Pose | x (m) | y (m) | z (m) | yaw (deg) | pitch (deg) | roll (deg) |",
                "|---|---|---|---|---|---|---|",
                row("Robot $A$ (world)", robot),
                row("Camera mount $B$ (robot frame)", camera_on_robot),
                row("Camera $A \\oplus B$ (world)", camera),
                row("$(A \\oplus B) \\ominus A$", recovered_mount),
                row("Inverse $A^{-1}$", robot_inv),
                row("$A \\oplus A^{-1}$", identity),
                "",
                f"Observed point: (2.5, 0, 0) in the camera frame, "
                f"**({point_world.x:.3f}, {point_world.y:.3f}, {point_world.z:.3f})** in the world "
                f"frame (`camera.composePoint()`), and ({point_robot.x:.3f}, {point_robot.y:.3f}, "
                f"{point_robot.z:.3f}) in the robot frame (`robot.inverseComposePoint()`).",
            ]
        )
    )
    return


@app.cell
def _(go, np):
    AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")  # x, y, z axes: red, green, blue

    def frame_traces(poses, length=1.0, width=6, opacity=1.0):
        """Three line traces (x, y, z axes) drawing the coordinate frames of a list of poses."""
        traces = []
        for axis, color in enumerate(AXIS_COLORS):
            pts = []
            for p in poses:
                o = np.array([p.x, p.y, p.z])
                tip = o + length * np.asarray(p.getRotationMatrix())[:, axis]
                pts += [o, tip, [None] * 3]
            pts = np.array(pts, dtype=object)
            traces.append(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines", opacity=opacity,
                    line=dict(color=color, width=width), hoverinfo="skip", showlegend=False,
                )
            )
        return traces

    def labels_trace(poses, names):
        """Text labels next to the origins of the given poses."""
        return go.Scatter3d(
            x=[p.x for p in poses], y=[p.y for p in poses], z=[p.z for p in poses],
            mode="text", text=names, textposition="bottom center", hoverinfo="skip", showlegend=False,
        )

    def scene_figure(traces, title):
        """A 3D figure with equal scales on all axes and some room around the data."""
        xyz = np.array([v for t in traces for v in zip(t.x, t.y, t.z) if v[0] is not None], dtype=float)
        center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        half = (xyz.max(axis=0) - xyz.min(axis=0)).max() / 2 + 0.5
        axes = {f"{a}axis": dict(range=[c - half, c + half], title=f"{a} (m)") for a, c in zip("xyz", center)}
        fig = go.Figure(traces)
        fig.update_layout(
            title=title, height=560, margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(aspectmode="cube", camera=dict(eye=dict(x=1.25, y=-1.25, z=0.7)), **axes),
        )
        return fig
    return frame_traces, labels_trace, scene_figure


@app.cell(hide_code=True)
def _(
    CPose3D,
    camera,
    frame_traces,
    go,
    labels_trace,
    point_world,
    robot,
    scene_figure,
):
    _world = CPose3D()
    _p = [point_world.x, point_world.y, point_world.z]
    scene_figure(
        [
            *frame_traces([_world], length=1.0),
            *frame_traces([robot], length=0.8),
            *frame_traces([camera], length=0.5),
            go.Scatter3d(
                x=[camera.x, _p[0]], y=[camera.y, _p[1]], z=[camera.z, _p[2]], mode="lines",
                line=dict(color="gray", width=3, dash="dash"), hoverinfo="skip", showlegend=False,
            ),
            go.Scatter3d(
                x=[_p[0]], y=[_p[1]], z=[_p[2]], mode="markers+text", text=["point"],
                textposition="bottom center", marker=dict(size=5, color="black"), showlegend=False,
            ),
            labels_trace([_world, robot, camera], ["world", "robot A", "camera A⊕B"]),
        ],
        "Frames: world, robot, and camera (x: red, y: green, z: blue)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Interpolating between poses

    Keyframe poses at known times are a common way to describe a trajectory.
    `CPose3DInterpolator` returns the pose at any time in between: linear
    interpolation for the translation and SLERP (spherical linear
    interpolation of quaternions) for the rotation, which turns at constant
    rate along the shortest arc. Here the keyframes are the world origin
    ($t = 0$ s), the robot pose $A$ ($t = 1$ s), and $A$ followed by a fixed
    motion ($t = 2$ s).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    t_query = mo.ui.slider(start=0, stop=2, step=0.05, value=1.4, label="Query time t (s)", show_value=True)
    t_query
    return (t_query,)


@app.cell
def _(CPose3D, CPose3DInterpolator, np, robot, t_query, timedelta):
    keyframes = [CPose3D(), robot, robot + CPose3D(2.0, 1.0, 0.5, np.radians(120), np.radians(-30), 0.0)]

    trajectory = CPose3DInterpolator()
    for _i, _kf in enumerate(keyframes):
        trajectory.insert(timedelta(seconds=_i), _kf.asTPose())  # time stamps: 0, 1, 2 s

    def pose_at(t):
        tpose, valid = trajectory.interpolate(timedelta(seconds=t))
        assert valid
        return CPose3D.fromTPose(tpose)

    trail = [pose_at(t) for t in np.linspace(0.0, 2.0, 21)]
    query = pose_at(t_query.value)
    return keyframes, query, trail


@app.cell(hide_code=True)
def _(frame_traces, go, keyframes, labels_trace, mo, np, query, scene_figure, t_query, trail):
    _path = np.array([[p.x, p.y, p.z] for p in trail])
    _fig = scene_figure(
        [
            go.Scatter3d(
                x=_path[:, 0], y=_path[:, 1], z=_path[:, 2], mode="lines",
                line=dict(color="gray", width=2), hoverinfo="skip", showlegend=False,
            ),
            *frame_traces(trail, length=0.25, width=3, opacity=0.5),
            *frame_traces(keyframes, length=0.7, width=6),
            *frame_traces([query], length=0.9, width=10),
            labels_trace([*keyframes, query], ["t = 0 s", "t = 1 s", "t = 2 s", f"t = {t_query.value:.2f} s"]),
        ],
        "Interpolated trajectory (thick: keyframes and the query time)",
    )
    _ypr = np.degrees([query.yaw, query.pitch, query.roll])
    mo.vstack(
        [
            mo.md(
                f"Pose at t = {t_query.value:.2f} s: x = {query.x:.3f}, y = {query.y:.3f}, "
                f"z = {query.z:.3f} m; yaw = {_ypr[0]:.1f}°, pitch = {_ypr[1]:.1f}°, roll = {_ypr[2]:.1f}°"
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/se3_poses.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
