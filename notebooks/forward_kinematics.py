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
    # Forward kinematics of robot arms

    **Forward (or direct) kinematics** answers: given the joint values
    $q = (q_1, \dots, q_n)$, where is the tool? With one Denavit-Hartenberg
    transformation $A_i(q_i)$ per link (see the
    [DH parameters notebook](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/dh_parameters.py)),
    the pose of the tool in the base frame is their composition:

    $$T_0^n(q) = A_1(q_1) \oplus A_2(q_2) \oplus \dots \oplus A_n(q_n)$$

    This notebook computes it with the [MRPT](https://www.mrpt.org/) Python
    wheel (`pip install mrpt3`), where every $A_i$ is an
    `mrpt.poses.CPose3D` and $\oplus$ is the `+` operator. It shows:

    1. The pose of the tool for any joint values, for six classic robots.
    2. The **workspace**: all the points the tool can reach.
    3. **Velocity kinematics**: the Jacobian, the velocity ellipsoid, and
       the **singular configurations** where the robot loses mobility.

    The [inverse kinematics notebook](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/inverse_kinematics.py)
    solves the opposite problem.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from mrpt.poses import CPose3D
    return CPose3D, go, mo, np


@app.cell
def _():
    # DH table rows: kind ("R" revolute / "P" prismatic), theta (deg), d (m), a (m), alpha (deg).
    # The joint variable replaces theta (R) or d (P). Limits and q in degrees (R) or meters (P).
    ROBOTS = {
        "Planar 3R arm": dict(
            links=[("R", 0, 0, 0.5, 0), ("R", 0, 0, 0.4, 0), ("R", 0, 0, 0.2, 0)],
            limits=[(-180, 180)] * 3, q=[30, 45, -40],
        ),
        "SCARA": dict(
            links=[("R", 0, 0.5, 0.45, 0), ("R", 0, 0, 0.35, 180), ("P", 0, 0, 0, 0), ("R", 0, 0.05, 0, 0)],
            limits=[(-150, 150), (-150, 150), (0, 0.3), (-180, 180)], q=[20, 60, 0.2, 0],
        ),
        "Cylindrical robot": dict(
            links=[("R", 0, 0.3, 0, 0), ("P", 0, 0, 0, -90), ("P", 0, 0, 0, 0)],
            limits=[(-180, 180), (0.1, 0.8), (0.1, 0.6)], q=[30, 0.5, 0.4],
        ),
        "Stanford arm": dict(
            links=[("R", 0, 0.4, 0, -90), ("R", 0, 0.15, 0, 90), ("P", 0, 0, 0, 0),
                   ("R", 0, 0, 0, -90), ("R", 0, 0, 0, 90), ("R", 0, 0.1, 0, 0)],
            limits=[(-180, 180), (-150, 150), (0.2, 0.8), (-180, 180), (-180, 180), (-180, 180)],
            q=[0, 60, 0.5, 0, 30, 0],
        ),
        "Anthropomorphic arm with spherical wrist": dict(
            links=[("R", 0, 0.4, 0, 90), ("R", 0, 0, 0.5, 0), ("R", 0, 0, 0, 90),
                   ("R", 0, 0.5, 0, -90), ("R", 0, 0, 0, 90), ("R", 0, 0.1, 0, 0)],
            limits=[(-180, 180)] * 6, q=[0, 45, 45, 0, 45, 0],
        ),
        "UR5": dict(
            links=[("R", 0, 0.089159, 0, 90), ("R", 0, 0, -0.425, 0), ("R", 0, 0, -0.39225, 0),
                   ("R", 0, 0.10915, 0, 90), ("R", 0, 0.09465, 0, -90), ("R", 0, 0.0823, 0, 0)],
            limits=[(-180, 180)] * 6, q=[0, -60, 90, -120, -90, 0],
        ),
    }
    return (ROBOTS,)


@app.cell
def _(CPose3D, np):
    def dh_pose(theta, d, a, alpha):
        """Pose of frame i in frame i-1: Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha). Angles in radians."""
        return CPose3D(0, 0, d, theta, 0, 0) + CPose3D(a, 0, 0, 0, 0, alpha)

    def link_params(link, q):
        """DH parameters (radians, meters) of a table row, with the joint variable q (rad or m)."""
        kind, theta, d, a, alpha = link
        theta, alpha = np.radians(theta), np.radians(alpha)
        if kind == "R":
            theta = q
        else:
            d = q
        return theta, d, a, alpha

    def forward_kinematics(links, q):
        """Poses of frames 0..n: the base (identity), then each one composed with the next link."""
        poses = [CPose3D()]
        for _link, _q in zip(links, q):
            poses.append(poses[-1] + dh_pose(*link_params(_link, _q)))
        return poses

    def to_joint_units(links, values):
        """Degrees to radians for the revolute joints; prismatic joints are already in meters."""
        return np.array([np.radians(v) if l[0] == "R" else v for l, v in zip(links, values)], dtype=float)

    def origin(p):
        return np.array([p.x, p.y, p.z])
    return forward_kinematics, link_params, origin, to_joint_units


@app.cell(hide_code=True)
def _(go, link_params, np, origin):
    AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")  # x, y, z axes: red, green, blue

    def frame_traces(poses, length=0.12, width=6, opacity=1.0):
        """Three line traces (x, y, z axes) drawing the coordinate frames of a list of poses."""
        traces = []
        for axis, color in enumerate(AXIS_COLORS):
            pts = []
            for p in poses:
                o = origin(p)
                pts += [o, o + length * np.asarray(p.getRotationMatrix())[:, axis], [None] * 3]
            pts = np.array(pts, dtype=object)
            traces.append(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines", opacity=opacity,
                    line=dict(color=color, width=width), hoverinfo="skip", showlegend=False,
                )
            )
        return traces

    def arm_traces(links, poses, q, opacity=1.0, color="#555555", frames=True):
        """The DH skeleton of the arm (d and a segments), its joints, and the frames."""
        pts = [origin(poses[0])]
        for i, link in enumerate(links):
            _, d, _, _ = link_params(link, q[i])
            pts += [origin(poses[i]) + d * np.asarray(poses[i].getRotationMatrix())[:, 2], origin(poses[i + 1])]
        pts = np.array(pts)
        traces = [
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines", opacity=opacity,
                line=dict(color=color, width=10), hoverinfo="skip", showlegend=False,
            )
        ]
        for kind, symbol in (("R", "circle"), ("P", "diamond")):
            js = np.array([origin(poses[i]) for i in range(len(links)) if links[i][0] == kind]).reshape(-1, 3)
            if len(js):
                traces.append(
                    go.Scatter3d(
                        x=js[:, 0], y=js[:, 1], z=js[:, 2], mode="markers", opacity=opacity,
                        marker=dict(size=6, symbol=symbol, color="#ffbf00", line=dict(color="black", width=1)),
                        hoverinfo="skip", showlegend=False,
                    )
                )
        if frames:
            traces += frame_traces(poses[:-1], opacity=opacity)
            traces += frame_traces(poses[-1:], length=0.2, width=8, opacity=opacity)  # the tool, larger
        return traces

    def scene_figure(traces, title, margin=0.15, height=560, bounds=None):
        """A 3D figure with equal scales on all axes; `bounds` (2x3) overrides the data extent."""
        if bounds is None:
            xyz = np.array(
                [
                    v
                    for t in traces
                    for v in zip(np.ravel(t.x), np.ravel(t.y), np.ravel(t.z))  # ravel: surfaces are 2D
                    if v[0] is not None and v[2] is not None
                ],
                dtype=float,
            )
            bounds = np.array([xyz.min(axis=0), xyz.max(axis=0)])
        center = bounds.mean(axis=0)
        half = (bounds[1] - bounds[0]).max() / 2 + margin
        axes = {f"{a}axis": dict(range=[c - half, c + half], title=f"{a} (m)") for a, c in zip("xyz", center)}
        fig = go.Figure(traces)
        fig.update_layout(
            title=title, height=height, margin=dict(l=0, r=0, t=40, b=0), legend=dict(x=0, y=1),
            scene=dict(aspectmode="cube", camera=dict(eye=dict(x=1.4, y=-1.2, z=0.8)), **axes),
            uirevision="keep",  # keep the user's 3D view while the sliders move
        )
        return fig
    return arm_traces, scene_figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. From joint values to the tool pose

    Choose a robot and **move the joint sliders**. The figure shows the DH
    skeleton of the arm (gray), its revolute (circles) and prismatic
    (diamonds) joints, the frame of every link and, larger, the tool frame
    $T_0^n$. Drag the figure to rotate it.
    """)
    return


@app.cell(hide_code=True)
def _(ROBOTS, mo):
    robot = mo.ui.dropdown(options=list(ROBOTS), value="Anthropomorphic arm with spherical wrist", label="Robot")
    robot
    return (robot,)


@app.cell(hide_code=True)
def _(ROBOTS, mo, robot):
    def joint_slider(i, kind, limits, value):
        if kind == "R":
            return mo.ui.slider(start=limits[0], stop=limits[1], step=1, value=value, label=f"θ{i} (deg)", show_value=True)
        return mo.ui.slider(start=limits[0], stop=limits[1], step=0.01, value=value, label=f"d{i} (m)", show_value=True)

    _r = ROBOTS[robot.value]
    joints = mo.ui.array(
        [joint_slider(i, l[0], lim, v) for i, (l, lim, v) in enumerate(zip(_r["links"], _r["limits"], _r["q"]), start=1)]
    )
    mo.hstack(joints.elements, wrap=True, justify="start", gap=1)
    return (joints,)


@app.cell
def _(ROBOTS, forward_kinematics, joints, robot, to_joint_units):
    links = ROBOTS[robot.value]["links"]
    q = to_joint_units(links, joints.value)
    poses = forward_kinematics(links, q)  # frames 0..n
    tool = poses[-1]
    return links, poses, q, tool


@app.cell(hide_code=True)
def _(arm_traces, links, mo, np, poses, q, robot, scene_figure, tool):
    def bmatrix(m):
        rows = [" & ".join(f"{v:+.3f}" for v in row) for row in np.round(m, 3) + 0.0]  # + 0.0: no "-0.000"
        return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"

    _xyz = np.round([tool.x, tool.y, tool.z], 3) + 0.0  # + 0.0: no "-0.000"
    _ypr = np.round(np.degrees([tool.yaw, tool.pitch, tool.roll]), 1) + 0.0
    mo.hstack(
        [
            scene_figure(arm_traces(links, poses, q), robot.value, height=520),
            mo.md(
                f"**Tool pose** $T_0^n$ `= poses[-1]`\n\n"
                f"| x (m) | y (m) | z (m) |\n|---|---|---|\n| {_xyz[0]:.3f} | {_xyz[1]:.3f} | {_xyz[2]:.3f} |\n\n"
                f"| yaw (deg) | pitch (deg) | roll (deg) |\n|---|---|---|\n"
                f"| {_ypr[0]:.1f} | {_ypr[1]:.1f} | {_ypr[2]:.1f} |\n\n"
                f"`tool.getHomogeneousMatrix()`:\n\n$${bmatrix(np.asarray(tool.getHomogeneousMatrix()))}$$\n\n"
                f"Its columns are the tool axes $x$, $y$, $z$ (the *approach* direction) and the position, "
                f"in the base frame."
            ),
        ],
        widths=[3, 2],
        align="center",
    )
    return (bmatrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The workspace

    The **workspace** is the set of all positions the tool can reach. A
    simple way to see it is to draw many random joint configurations, within
    the joint limits, and plot where the tool ends up. Its shape gives the
    robot family its name: an annulus for the planar arm, a cylinder shell
    for the cylindrical robot, a thick sphere for the anthropomorphic arm...

    The points are colored by the **manipulability** of their configuration
    (section 3): it drops near the borders of the workspace, where the arm is
    stretched or folded.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_samples = mo.ui.slider(start=500, stop=5000, step=500, value=2000, label="Random configurations", show_value=True)
    n_samples
    return (n_samples,)


@app.cell
def _(np, origin):
    def jacobian(links, poses):
        """Geometric Jacobian (6 x n): tool linear velocity (rows 0-2) and angular velocity (rows 3-5)."""
        o_n = origin(poses[-1])
        J = np.zeros((6, len(links)))
        for i, link in enumerate(links):
            z = np.asarray(poses[i].getRotationMatrix())[:, 2]  # axis of joint i+1
            if link[0] == "R":
                J[:3, i] = np.cross(z, o_n - origin(poses[i]))
                J[3:, i] = z
            else:
                J[:3, i] = z
        return J

    def task_rank(J, tol=1e-6):
        """Number of independent tool velocities (rank of J), ignoring tiny singular values."""
        s = np.linalg.svd(J, compute_uv=False)
        return int((s > tol * max(s[0], 1e-12)).sum())

    def manipulability(J, rank):
        """Yoshikawa's measure: the product of the `rank` largest singular values (the volume of the ellipsoid)."""
        return float(np.prod(np.linalg.svd(J, compute_uv=False)[:rank]))
    return jacobian, manipulability, task_rank


@app.cell
def _(
    ROBOTS,
    forward_kinematics,
    jacobian,
    links,
    manipulability,
    n_samples,
    np,
    origin,
    robot,
    task_rank,
    to_joint_units,
):
    _rng = np.random.default_rng(0)
    _lim = np.array(ROBOTS[robot.value]["limits"], dtype=float)
    _samples = to_joint_units(links, _lim[:, 0]) + _rng.random((n_samples.value, len(links))) * (
        to_joint_units(links, _lim[:, 1]) - to_joint_units(links, _lim[:, 0])
    )
    # The rank of the Jacobian at a random configuration (the generic one):
    generic_rank = task_rank(jacobian(links, forward_kinematics(links, _samples[0])))

    ws_points, ws_manip = [], []
    for _qs in _samples:
        _poses = forward_kinematics(links, _qs)
        ws_points.append(origin(_poses[-1]))
        ws_manip.append(manipulability(jacobian(links, _poses), generic_rank))
    ws_points, ws_manip = np.array(ws_points), np.array(ws_manip)
    return generic_rank, ws_manip, ws_points


@app.cell(hide_code=True)
def _(
    arm_traces,
    go,
    links,
    np,
    poses,
    q,
    robot,
    scene_figure,
    ws_manip,
    ws_points,
):
    _reach = np.linalg.norm(ws_points - np.array([poses[0].x, poses[0].y, poses[0].z]), axis=1)
    scene_figure(
        [
            go.Scatter3d(
                x=ws_points[:, 0], y=ws_points[:, 1], z=ws_points[:, 2], mode="markers",
                marker=dict(size=2, color=ws_manip, colorscale="Viridis", colorbar=dict(title="manip.", x=1.0)),
                hoverinfo="skip", showlegend=False,
            ),
            *arm_traces(links, poses, q, frames=False),
        ],
        f"{robot.value}: {len(ws_points)} reachable tool positions (farthest at {_reach.max():.2f} m)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Velocity kinematics and singularities

    How fast does the tool move when the joints move? Differentiating the
    forward kinematics gives a linear relation, through the **geometric
    Jacobian** $J(q)$, a $6 \times n$ matrix:

    $$\begin{bmatrix} v \\ \omega \end{bmatrix} = J(q)\, \dot q, \qquad
    J_i = \begin{cases}
    \begin{bmatrix} z_{i-1} \times (o_n - o_{i-1}) \\ z_{i-1} \end{bmatrix} & \text{revolute joint} \\[2mm]
    \begin{bmatrix} z_{i-1} \\ 0 \end{bmatrix} & \text{prismatic joint}
    \end{cases}$$

    where $v$ and $\omega$ are the linear and angular velocities of the tool,
    and $z_{i-1}$, $o_{i-1}$ the axis and origin of frame $i-1$, all read from
    the `CPose3D` frames of section 1. Below, $J$ is compared with numerical
    derivatives of the forward kinematics.

    The **singular values** of $J$ tell how easily the tool moves in each
    direction. At a **singular configuration** some of them vanish: the rank
    of $J$ drops and the tool cannot move in some direction, whatever the
    joint speeds. The **velocity ellipsoid**, the set of tool velocities $v$
    reachable with $\|\dot q\| \le 1$, then collapses into a flat shape.
    **Try these singularities** with the sliders of section 1:

    - Planar 3R arm: $\theta_2 = 0°$, the first two links aligned.
    - Anthropomorphic arm: $\theta_5 = 0°$ (wrist: axes 4 and 6 aligned) or
      $\theta_3 = 90°$ (elbow: arm fully stretched).
    - UR5: $\theta_5 = 0°$ (wrist) or $\theta_3 = 0°$ (elbow).
    """)
    return


@app.cell
def _(forward_kinematics, jacobian, links, np, poses, q, tool):
    def rotation_log(R):
        """Rotation vector (axis times angle) of a rotation matrix, for small rotations."""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
        return w if angle < 1e-9 else w * angle / np.sin(angle)

    J = jacobian(links, poses)

    # Numerical derivatives: move one joint a tiny step and see how the tool pose changes.
    _h = 1e-6
    J_num = np.zeros_like(J)
    _R0 = np.asarray(tool.getRotationMatrix())
    for _i in range(len(links)):
        _dq = q.copy()
        _dq[_i] += _h
        _t = forward_kinematics(links, _dq)[-1]
        J_num[:3, _i] = (np.array([_t.x - tool.x, _t.y - tool.y, _t.z - tool.z])) / _h
        J_num[3:, _i] = rotation_log(np.asarray(_t.getRotationMatrix()) @ _R0.T) / _h

    U, sigma, _ = np.linalg.svd(J)
    Uv, sigma_v, _ = np.linalg.svd(J[:3])  # linear velocity part, for the ellipsoid
    return J, J_num, U, Uv, sigma, sigma_v


@app.cell(hide_code=True)
def _(J, J_num, U, generic_rank, manipulability, mo, np, sigma):
    def jmatrix(m):
        names = [r"v_x", r"v_y", r"v_z", r"\omega_x", r"\omega_y", r"\omega_z"]
        rows = [n + " & " + " & ".join(f"{v:+.3f}" for v in row) for n, row in zip(names, np.round(m, 3) + 0.0)]
        cols = " & ".join(rf"\dot q_{i}" for i in range(1, m.shape[1] + 1))
        return r"\begin{array}{r|" + "r" * m.shape[1] + "} & " + cols + r" \\ \hline " + r" \\ ".join(rows) + r"\end{array}"

    _r = generic_rank
    _ratio = sigma[_r - 1] / sigma[0]
    _singular = _ratio < 1e-3
    _lost = U[:, _r - 1]
    _what = "move along" if np.linalg.norm(_lost[:3]) > np.linalg.norm(_lost[3:]) else "rotate about"
    _vec = _lost[:3] if _what == "move along" else _lost[3:]
    _vec = _vec / np.linalg.norm(_vec)
    _status = (
        mo.callout(
            mo.md(
                f"**Singular configuration**: the rank of $J$ drops from {_r} to "
                f"{int((sigma > 1e-3 * sigma[0]).sum())}. The tool cannot {_what} the direction "
                f"({_vec[0]:+.2f}, {_vec[1]:+.2f}, {_vec[2]:+.2f}) of the base frame."
            ),
            kind="warn",
        )
        if _singular
        else mo.md(
            f"Not singular: the smallest relevant singular value is {_ratio:.2f} times the largest one."
        )
    )
    mo.vstack(
        [
            mo.md(f"$$J = {jmatrix(J)}$$"),
            mo.md(
                f"Largest difference with the numerical derivatives: {np.abs(J - J_num).max():.1e}. "
                f"This robot can produce at most **{_r} independent tool velocities** (the rank of $J$ at a "
                f"generic configuration); singular values: "
                + ", ".join(f"{s:.3f}" for s in sigma[:_r])
                + f"; manipulability $w = \\prod \\sigma_i$ = **{manipulability(J, _r):.4f}**."
            ),
            _status,
        ]
    )
    return


@app.cell(hide_code=True)
def _(Uv, arm_traces, go, links, np, poses, q, scene_figure, sigma_v, tool):
    # Ellipsoid of tool linear velocities for unit joint speed: axes U_i scaled by sigma_i.
    _u, _v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
    _sphere = np.stack([np.cos(_u) * np.sin(_v), np.sin(_u) * np.sin(_v), np.cos(_v)])
    _scale = 0.3 / max(sigma_v[0], 1e-9)  # draw the largest axis 0.3 m long
    _e = np.einsum("ij,jkl->ikl", Uv @ np.diag(sigma_v * _scale), _sphere)
    _o = np.array([tool.x, tool.y, tool.z])
    scene_figure(
        [
            *arm_traces(links, poses, q, frames=False),
            go.Surface(
                x=_o[0] + _e[0], y=_o[1] + _e[1], z=_o[2] + _e[2], opacity=0.45, showscale=False,
                colorscale=[[0, "#1f77b4"], [1, "#1f77b4"]], hoverinfo="skip",
            ),
        ],
        "Tool velocity ellipsoid (largest axis drawn 0.3 m long)",
        height=480,
    )
    return


if __name__ == "__main__":
    app.run()
