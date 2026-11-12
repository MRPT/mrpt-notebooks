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
    # Denavit-Hartenberg parameters

    A robot arm is a chain of rigid links joined by joints. To know where its
    tool is, we attach a coordinate frame to every link and write the pose of
    each frame relative to the previous one. A general 3D pose needs six
    numbers, but the **Denavit-Hartenberg (DH) convention** places the frames
    so cleverly that **four numbers per link** are enough:
    $\theta_i$, $d_i$, $a_i$ and $\alpha_i$.

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`): each link transformation is an
    `mrpt.poses.CPose3D`, and chaining links is pose composition $\oplus$ (the
    `+` operator). It shows:

    1. The four elementary motions of one link, step by step.
    2. The rules to place the frames, and how to read the parameters off a
       drawing.
    3. The DH tables of classic manipulators, drawn frame by frame.
    4. An editable table to check your own exercises.

    Next steps:
    [forward kinematics](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/forward_kinematics.py)
    and
    [inverse kinematics](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/inverse_kinematics.py).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from mrpt.poses import CPose3D
    return CPose3D, go, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One link: four elementary motions

    In the standard DH convention, frame $i$ is reached from frame $i-1$ with
    four motions, each one along or about an axis of the frame obtained so
    far:

    $$A_i = \mathrm{Rot}_z(\theta_i)\; \mathrm{Trans}_z(d_i)\; \mathrm{Trans}_x(a_i)\; \mathrm{Rot}_x(\alpha_i)$$

    1. Rotate $\theta_i$ about $z_{i-1}$: the **joint angle**.
    2. Move $d_i$ along $z_{i-1}$: the **link offset**.
    3. Move $a_i$ along the new $x$ axis, which is already $x_i$: the
       **link length**.
    4. Rotate $\alpha_i$ about $x_i$: the **link twist**, which turns the
       old $z$ axis into $z_i$.

    The first two motions commute, and so do the last two, so in MRPT a link
    is the composition of just two poses, each one a translation plus a
    rotation:
    `CPose3D(0, 0, d, θ, 0, 0) + CPose3D(a, 0, 0, 0, 0, α)` (the arguments are
    x, y, z, yaw, pitch, roll; yaw turns about $z$ and roll about $x$).

    **Move the sliders**, and use the last one to apply the motions one at a
    time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    theta = mo.ui.slider(start=-180, stop=180, step=5, value=40, label="θ (deg)", show_value=True)
    d = mo.ui.slider(start=-1, stop=1, step=0.05, value=0.4, label="d (m)", show_value=True)
    a = mo.ui.slider(start=-1, stop=1, step=0.05, value=0.8, label="a (m)", show_value=True)
    alpha = mo.ui.slider(start=-180, stop=180, step=5, value=60, label="α (deg)", show_value=True)
    n_motions = mo.ui.slider(start=0, stop=4, step=1, value=4, label="Motions applied", show_value=True)
    mo.vstack([mo.hstack([theta, d, a, alpha], justify="start", gap=2), n_motions])
    return a, alpha, d, n_motions, theta


@app.cell
def _(CPose3D, np):
    def dh_pose(theta, d, a, alpha):
        """Pose of frame i in frame i-1: Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha). Angles in radians."""
        return CPose3D(0, 0, d, theta, 0, 0) + CPose3D(a, 0, 0, 0, 0, alpha)

    def dh_matrix(theta, d, a, alpha):
        """The same transformation, as the closed-form 4x4 homogeneous matrix of the textbooks."""
        ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return dh_matrix, dh_pose


@app.cell
def _(CPose3D, a, alpha, d, dh_matrix, dh_pose, np, theta):
    _th, _al = np.radians(theta.value), np.radians(alpha.value)
    elementary = [
        ("Rot_z(θ)", CPose3D(0, 0, 0, _th, 0, 0)),
        ("Trans_z(d)", CPose3D(0, 0, d.value, 0, 0, 0)),
        ("Trans_x(a)", CPose3D(a.value, 0, 0, 0, 0, 0)),
        ("Rot_x(α)", CPose3D(0, 0, 0, 0, 0, _al)),
    ]
    # The frame after each elementary motion, composed one at a time:
    partial = [CPose3D()]
    for _name, _motion in elementary:
        partial.append(partial[-1] + _motion)

    link = dh_pose(_th, d.value, a.value, _al)
    A_mrpt = np.asarray(link.getHomogeneousMatrix())
    A_formula = dh_matrix(_th, d.value, a.value, _al)
    return A_formula, A_mrpt, elementary, partial


@app.cell(hide_code=True)
def _(A_formula, A_mrpt, mo, np, partial):
    def bmatrix(m):
        rows = [" & ".join(f"{v:+.3f}" for v in row) for row in np.round(m, 3) + 0.0]  # + 0.0: no "-0.000"
        return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"

    mo.hstack(
        [
            mo.md(
                r"""
    **Closed form** (with $c_\theta = \cos\theta_i$, $s_\alpha = \sin\alpha_i$...)

    $$A_i = \begin{bmatrix}
    c_\theta & -s_\theta c_\alpha & s_\theta s_\alpha & a_i c_\theta \\
    s_\theta & c_\theta c_\alpha & -c_\theta s_\alpha & a_i s_\theta \\
    0 & s_\alpha & c_\alpha & d_i \\
    0 & 0 & 0 & 1
    \end{bmatrix}$$
    """
            ),
            mo.md(
                f"**From MRPT** `dh_pose(...).getHomogeneousMatrix()`\n\n$$A_i = {bmatrix(A_mrpt)}$$\n\n"
                f"Largest difference with the closed form: {np.abs(A_mrpt - A_formula).max():.1e}. "
                f"The last motion, alone, gives the same frame: "
                f"{np.abs(np.asarray(partial[4].getHomogeneousMatrix()) - A_mrpt).max():.1e}."
            ),
        ],
        widths="equal",
        gap=3,
    )
    return (bmatrix,)


@app.cell(hide_code=True)
def _(go, np):
    AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")  # x, y, z axes: red, green, blue

    def origin(p):
        return np.array([p.x, p.y, p.z])

    def frame_traces(poses, length=0.3, width=6, opacity=1.0):
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

    def labels_trace(points, names, color="black"):
        """Text labels at the given 3D points."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="text", text=names,
            textposition="top center", textfont=dict(color=color), hoverinfo="skip", showlegend=False,
        )

    def segment_trace(points, color, width=8, dash="solid", name=None):
        """A polyline through the given 3D points (shown in the legend if it has a name)."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash), hoverinfo="skip", showlegend=name is not None,
        )

    def scene_figure(traces, title, margin=0.3, height=560):
        """A 3D figure with equal scales on all axes and some room around the data."""
        xyz = np.array(
            [v for t in traces for v in zip(t.x, t.y, t.z) if v[0] is not None and v[2] is not None], dtype=float
        )
        center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        half = (xyz.max(axis=0) - xyz.min(axis=0)).max() / 2 + margin
        axes = {f"{a}axis": dict(range=[c - half, c + half], title=f"{a} (m)") for a, c in zip("xyz", center)}
        fig = go.Figure(traces)
        fig.update_layout(
            title=title, height=height, margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(x=0, y=1), scene=dict(aspectmode="cube", camera=dict(eye=dict(x=1.4, y=-1.2, z=0.8)), **axes),
        )
        return fig
    return frame_traces, labels_trace, origin, scene_figure, segment_trace


@app.cell(hide_code=True)
def _(
    a,
    d,
    elementary,
    frame_traces,
    labels_trace,
    n_motions,
    np,
    origin,
    partial,
    scene_figure,
    segment_trace,
):
    _k = n_motions.value
    _span = max(abs(d.value), abs(a.value)) + 0.4  # length of the joint axes drawn
    _traces = [
        segment_trace([[0, 0, -_span], [0, 0, _span]], "gray", width=3, dash="dash", name="joint axis z(i-1)"),
        *frame_traces([partial[0]], length=0.4),
    ]
    if _k >= 2:
        _traces.append(segment_trace([origin(partial[0]), origin(partial[2])], "#9467bd", name="offset d"))
    if _k >= 3:
        _traces.append(segment_trace([origin(partial[2]), origin(partial[3])], "#ff7f0e", name="common normal, length a"))
    if _k == 4:
        _zi = np.asarray(partial[4].getRotationMatrix())[:, 2]
        _oi = origin(partial[4])
        _traces.append(
            segment_trace([_oi - _span * _zi, _oi + _span * _zi], "black", width=3, dash="dash", name="next joint axis z(i)")
        )
    # Intermediate frames, faded; the current one, solid:
    if _k > 0:
        _traces += frame_traces(partial[1:_k], length=0.3, opacity=0.3)
        _traces += frame_traces([partial[_k]], length=0.4)
    # Labels for the first frame and the current one (the others are faded):
    _traces.append(labels_trace([origin(partial[0])], ["frame i-1"]))
    if _k > 0:
        _label = "frame i" if _k == 4 else f"after {elementary[_k - 1][0]}"
        _traces.append(labels_trace([origin(partial[_k]) + [0, 0, 0.05]], [_label], color="#d62728"))
    scene_figure(_traces, f"{_k} of 4 motions applied (axes x: red, y: green, z: blue)", margin=0.1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Placing the frames

    Four numbers are enough only because each frame is placed following some
    rules. For a chain with joints $1, \dots, n$, frame $0$ is the base and
    frame $i$ is attached to link $i$, the one moved by joint $i$:

    - $z_{i-1}$ lies along the axis of joint $i$: the rotation axis of a
      revolute joint, or the sliding direction of a prismatic joint.
    - $x_i$ lies along the **common normal** of $z_{i-1}$ and $z_i$: the
      line perpendicular to both, pointing from $z_{i-1}$ towards $z_i$. The
      origin $o_i$ is where it meets $z_i$.
    - $y_i$ completes a right-handed frame.

    With the frames in place, the parameters can be read off a drawing (in
    the figure above, the dashed lines are the joint axes):

    | Parameter | Meaning | Measured |
    |---|---|---|
    | $a_i$, link length | distance between $z_{i-1}$ and $z_i$ (orange) | along $x_i$ |
    | $\alpha_i$, link twist | angle from $z_{i-1}$ to $z_i$ | about $x_i$ |
    | $d_i$, link offset | distance from $o_{i-1}$ to the common normal (purple) | along $z_{i-1}$ |
    | $\theta_i$, joint angle | angle from $x_{i-1}$ to $x_i$ | about $z_{i-1}$ |

    $a_i$ and $\alpha_i$ describe the shape of the link and never change. Of
    $d_i$ and $\theta_i$, one is the **joint variable** $q_i$ ($\theta_i$ for
    a revolute joint, $d_i$ for a prismatic one) and the other is constant.

    The rules leave some free choices:

    - If $z_{i-1}$ and $z_i$ are **parallel**, there are infinitely many
      common normals: pick the one that makes $d_i = 0$.
    - If they **intersect**, $a_i = 0$ and $x_i$ is perpendicular to the
      plane of both axes (either direction).
    - The base frame only needs $z_0$ along the first joint axis, and the
      last frame (the tool) is chosen freely.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. DH tables of classic manipulators

    Below are the DH tables of six robots, with lengths in meters and angles
    in degrees. $q_i$ marks the joint variables, shown at a sample
    configuration. **Move the slider** to build the robot frame by frame: the
    gray line is the chain of $d_i$ and $a_i$ segments, and the circles
    (revolute) and diamonds (prismatic) mark the joints, on the $z$ axis of
    the previous frame.
    """)
    return


@app.cell
def _():
    # kind ("R" revolute / "P" prismatic), theta (deg), d (m), a (m), alpha (deg).
    # The joint variable replaces theta (R) or d (P). "q" is a sample configuration (deg or m).
    ROBOTS = {
        "Planar 3R arm": dict(
            links=[("R", 0, 0, 0.5, 0), ("R", 0, 0, 0.4, 0), ("R", 0, 0, 0.2, 0)],
            q=[30, 45, -40],
            about="All joint axes are parallel (vertical), so every twist is zero and the offsets can be set to zero.",
        ),
        "SCARA": dict(
            links=[("R", 0, 0.5, 0.45, 0), ("R", 0, 0, 0.35, 180), ("P", 0, 0, 0, 0), ("R", 0, 0.05, 0, 0)],
            q=[20, 60, 0.2, 0],
            about="Two vertical revolute joints move the arm in a horizontal plane; then a vertical prismatic "
            "joint and a rotation of the tool. The twist of 180° in link 2 makes $z_2$ point down, so a positive "
            "$d_3$ lowers the tool.",
        ),
        "Cylindrical robot": dict(
            links=[("R", 0, 0.3, 0, 0), ("P", 0, 0, 0, -90), ("P", 0, 0, 0, 0)],
            q=[30, 0.5, 0.4],
            about="A vertical rotation, then a vertical and a horizontal prismatic joint: the tool moves in "
            "cylindrical coordinates $(\\theta_1, d_2, d_3)$.",
        ),
        "Stanford arm": dict(
            links=[("R", 0, 0.4, 0, -90), ("R", 0, 0.15, 0, 90), ("P", 0, 0, 0, 0),
                   ("R", 0, 0, 0, -90), ("R", 0, 0, 0, 90), ("R", 0, 0.1, 0, 0)],
            q=[0, 60, 0.5, 0, 30, 0],
            about="A spherical arm (two rotations and a prismatic joint) followed by a spherical wrist: the "
            "axes of joints 4, 5 and 6 meet at one point, so $a_4 = a_5 = d_5 = 0$.",
        ),
        "Anthropomorphic arm with spherical wrist": dict(
            links=[("R", 0, 0.4, 0, 90), ("R", 0, 0, 0.5, 0), ("R", 0, 0, 0, 90),
                   ("R", 0, 0.5, 0, -90), ("R", 0, 0, 0, 90), ("R", 0, 0.1, 0, 0)],
            q=[0, 45, 45, 0, 45, 0],
            about="Like a human arm: a shoulder with two axes, an elbow parallel to the second one, and a "
            "spherical wrist. The forearm lies along $z_3$, so its length is $d_4$ and not an $a_i$.",
        ),
        "UR5": dict(
            links=[("R", 0, 0.089159, 0, 90), ("R", 0, 0, -0.425, 0), ("R", 0, 0, -0.39225, 0),
                   ("R", 0, 0.10915, 0, 90), ("R", 0, 0.09465, 0, -90), ("R", 0, 0.0823, 0, 0)],
            q=[0, -90, 0, -90, 0, 0],
            about="The DH table published by Universal Robots for this collaborative robot. Its wrist is not "
            "spherical: the offsets $d_4$, $d_5$ and $d_6$ separate the last three axes.",
        ),
    }
    return (ROBOTS,)


@app.cell
def _(CPose3D, dh_pose, np):
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
    return forward_kinematics, link_params, to_joint_units


@app.cell(hide_code=True)
def _(frame_traces, go, labels_trace, link_params, np, origin, segment_trace):
    def arm_traces(links, poses, q, n_frames=None, frame_length=0.12, opacity=1.0, color="#555555"):
        """The DH skeleton of the arm (d and a segments), its joints, and frames 0..n_frames."""
        n = len(links) if n_frames is None else n_frames
        pts = [origin(poses[0])]
        for i in range(n):
            _, d, _, _ = link_params(links[i], q[i])
            z = np.asarray(poses[i].getRotationMatrix())[:, 2]
            pts += [origin(poses[i]) + d * z, origin(poses[i + 1])]
        traces = [segment_trace(pts, color, width=10)] if n > 0 else []
        for kind, symbol in (("R", "circle"), ("P", "diamond")):
            js = [origin(poses[i]) for i in range(n) if links[i][0] == kind]
            if js:
                js = np.array(js)
                traces.append(
                    go.Scatter3d(
                        x=js[:, 0], y=js[:, 1], z=js[:, 2], mode="markers",
                        marker=dict(size=7, symbol=symbol, color="#ffbf00", line=dict(color="black", width=1)),
                        hoverinfo="skip", showlegend=False,
                    )
                )
        for t in traces:
            t.opacity = opacity
        traces += frame_traces(poses[: n + 1], length=frame_length, opacity=opacity)
        return traces

    def frame_labels(poses, n_frames):
        return labels_trace([origin(p) for p in poses[: n_frames + 1]], [str(i) for i in range(n_frames + 1)])
    return arm_traces, frame_labels


@app.cell(hide_code=True)
def _(ROBOTS, mo):
    robot = mo.ui.dropdown(options=list(ROBOTS), value="Anthropomorphic arm with spherical wrist", label="Robot")
    robot
    return (robot,)


@app.cell(hide_code=True)
def _(ROBOTS, mo, robot):
    n_frames = mo.ui.slider(
        start=0, stop=len(ROBOTS[robot.value]["links"]), step=1,
        value=len(ROBOTS[robot.value]["links"]), label="Frames shown", show_value=True,
    )
    n_frames
    return (n_frames,)


@app.cell(hide_code=True)
def _(ROBOTS, mo, robot):
    def dh_table(links, q):
        """Markdown DH table; joint variables as q_i with their value."""
        rows = ["| Link $i$ | Joint | $\\theta_i$ (deg) | $d_i$ (m) | $a_i$ (m) | $\\alpha_i$ (deg) |", "|---|---|---|---|---|---|"]
        for i, ((kind, theta, d, a, alpha), qi) in enumerate(zip(links, q), start=1):
            th = f"$q_{i}$ = {qi:g}" if kind == "R" else f"{theta:g}"
            dd = f"$q_{i}$ = {qi:g}" if kind == "P" else f"{d:g}"
            rows.append(f"| {i} | {'revolute' if kind == 'R' else 'prismatic'} | {th} | {dd} | {a:g} | {alpha:g} |")
        return "\n".join(rows)

    _r = ROBOTS[robot.value]
    mo.md(dh_table(_r["links"], _r["q"]) + "\n\n" + _r["about"])
    return (dh_table,)


@app.cell(hide_code=True)
def _(
    ROBOTS,
    arm_traces,
    forward_kinematics,
    frame_labels,
    n_frames,
    robot,
    scene_figure,
    to_joint_units,
):
    _r = ROBOTS[robot.value]
    _q = to_joint_units(_r["links"], _r["q"])
    _poses = forward_kinematics(_r["links"], _q)
    _n = n_frames.value
    scene_figure(
        arm_traces(_r["links"], _poses, _q, n_frames=_n) + [frame_labels(_poses, _n)],
        f"{robot.value}: frames 0 to {_n}",
        margin=0.15,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Your own DH table

    Write one link per line: the joint type (`R` or `P`), then $\theta$
    (deg), $d$ (m), $a$ (m) and $\alpha$ (deg), separated by commas. For the
    joint variable ($\theta$ of `R`, $d$ of `P`) write the value to draw.
    Lines starting with `#` are comments.

    Some exercises:

    - A planar 2R arm with links of 1 m and 0.5 m: check that with
      $\theta = (0°, 90°)$ the tool is at $(1, 0.5, 0)$.
    - A Cartesian robot (three prismatic joints along $z$, $x$ and $y$):
      which twists $\alpha_i$ make the three axes perpendicular?
    - Change the cylindrical robot of section 3 so that its last joint is
      vertical too. Is it still a useful robot?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    user_table = mo.ui.text_area(
        value="# type, theta, d, a, alpha\nR, 30, 0.5, 0, 90\nR, 20, 0, 0.6, 0\nR, -40, 0, 0.4, 0\nP, 0, 0.1, 0, 0",
        rows=8, label="DH table", full_width=True,
    )
    user_table
    return (user_table,)


@app.cell
def _(user_table):
    def parse_dh_table(text):
        """List of (kind, theta, d, a, alpha) rows and the joint values, or an error message."""
        links, q = [], []
        for n, line in enumerate(text.splitlines(), start=1):
            line = line.split("#")[0].strip()
            if not line:
                continue
            fields = [f.strip() for f in line.split(",")]
            if len(fields) != 5 or fields[0].upper() not in ("R", "P"):
                return None, None, f"Line {n}: expected `R` or `P` and four numbers, got `{line}`"
            try:
                theta, d, a, alpha = (float(f) for f in fields[1:])
            except ValueError:
                return None, None, f"Line {n}: `{line}` has a field that is not a number"
            kind = fields[0].upper()
            links.append((kind, theta, d, a, alpha))
            q.append(theta if kind == "R" else d)
        if not links:
            return None, None, "The table is empty"
        return links, q, None

    user_links, user_q, user_error = parse_dh_table(user_table.value)
    return user_error, user_links, user_q


@app.cell(hide_code=True)
def _(
    arm_traces,
    bmatrix,
    dh_table,
    forward_kinematics,
    frame_labels,
    mo,
    np,
    scene_figure,
    to_joint_units,
    user_error,
    user_links,
    user_q,
):
    if user_error:
        _out = mo.callout(mo.md(user_error), kind="danger")
    else:
        _q = to_joint_units(user_links, user_q)
        _poses = forward_kinematics(user_links, _q)
        _tool = _poses[-1]
        _out = mo.vstack(
            [
                mo.md(dh_table(user_links, user_q)),
                mo.md(
                    f"Tool frame in the base frame, $T = A_1 \\oplus \\dots \\oplus A_{len(user_links)}$:\n\n"
                    f"$${bmatrix(np.asarray(_tool.getHomogeneousMatrix()))}$$"
                ),
                scene_figure(
                    arm_traces(user_links, _poses, _q) + [frame_labels(_poses, len(user_links))],
                    "Your robot", margin=0.15,
                ),
            ]
        )
    _out
    return


if __name__ == "__main__":
    app.run()
