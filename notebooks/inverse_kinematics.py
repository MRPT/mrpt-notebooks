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
    # Inverse kinematics of robot arms

    **Inverse kinematics** (IK) is the opposite of
    [forward kinematics](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/forward_kinematics.py):
    given a desired pose of the tool, find joint values $q$ that reach it.
    It is harder: the equations are nonlinear, and there may be several
    solutions, infinitely many, or none at all.

    This notebook solves it in three ways, using the
    [MRPT](https://www.mrpt.org/) Python wheel (`pip install mrpt3`) for the
    poses (`mrpt.poses.CPose3D`) and the forward kinematics (pose
    composition $\oplus$, the `+` operator), which checks every solution:

    1. **Closed form** for a planar arm with two links: two solutions.
    2. **Kinematic decoupling** for a 6-DOF arm with a spherical wrist:
       eight solutions.
    3. **Numerical methods** based on the Jacobian, which work for any
       robot: Jacobian transpose, pseudo-inverse and damped least squares.

    The robots are described with Denavit-Hartenberg tables (see the
    [DH parameters notebook](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/dh_parameters.py)).
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

    def origin(p):
        return np.array([p.x, p.y, p.z])

    def rotation_log(R):
        """Rotation vector (unit axis times angle, radians) of a rotation matrix."""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle > np.pi - 1e-6:  # 180 deg: the axis is the dominant column of R + I
            k = np.argmax(np.diag(R))
            v = R[:, k] + np.eye(3)[k]
            return angle * v / np.linalg.norm(v)
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
        return w if angle < 1e-9 else w * angle / np.sin(angle)

    def pose_error(target, current):
        """6-vector (position error, rotation vector) from `current` to `target`, in the base frame."""
        R_err = np.asarray(target.getRotationMatrix()) @ np.asarray(current.getRotationMatrix()).T
        return np.concatenate([origin(target) - origin(current), rotation_log(R_err)])
    return forward_kinematics, link_params, origin, pose_error


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

    def arm_traces(links, poses, q, opacity=1.0, color="#555555", name=None, width=10):
        """The DH skeleton of the arm (d and a segments) and its joints."""
        pts = [origin(poses[0])]
        for i, link in enumerate(links):
            _, d, _, _ = link_params(link, q[i])
            pts += [origin(poses[i]) + d * np.asarray(poses[i].getRotationMatrix())[:, 2], origin(poses[i + 1])]
        pts = np.array(pts)
        joints = np.array([origin(p) for p in poses[:-1]])
        return [
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines", opacity=opacity, name=name,
                line=dict(color=color, width=width), hoverinfo="skip", showlegend=name is not None,
            ),
            go.Scatter3d(
                x=joints[:, 0], y=joints[:, 1], z=joints[:, 2], mode="markers", opacity=opacity,
                marker=dict(size=4, color="#ffbf00", line=dict(color="black", width=1)),
                hoverinfo="skip", showlegend=False,
            ),
        ]

    def scene_figure(traces, title, margin=0.15, height=560):
        """A 3D figure with equal scales on all axes and some room around the data."""
        xyz = np.array(
            [v for t in traces for v in zip(t.x, t.y, t.z) if v[0] is not None and v[2] is not None], dtype=float
        )
        center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        half = (xyz.max(axis=0) - xyz.min(axis=0)).max() / 2 + margin
        axes = {f"{a}axis": dict(range=[c - half, c + half], title=f"{a} (m)") for a, c in zip("xyz", center)}
        fig = go.Figure(traces)
        fig.update_layout(
            title=title, height=height, margin=dict(l=0, r=0, t=40, b=0), legend=dict(x=0, y=1),
            scene=dict(aspectmode="cube", camera=dict(eye=dict(x=1.4, y=-1.2, z=0.8)), **axes),
            uirevision="keep",  # keep the user's 3D view while the sliders move
        )
        return fig
    return arm_traces, frame_traces, scene_figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Closed form: a planar arm with two links

    The tool of a planar arm with links of lengths $l_1$ and $l_2$ is at

    $$x = l_1 \cos\theta_1 + l_2 \cos(\theta_1 + \theta_2), \qquad
    y = l_1 \sin\theta_1 + l_2 \sin(\theta_1 + \theta_2).$$

    Squaring and adding both equations removes $\theta_1$ (the law of
    cosines), and then $\theta_1$ follows from the angle to the target:

    $$\cos\theta_2 = \frac{x^2 + y^2 - l_1^2 - l_2^2}{2\, l_1 l_2}, \qquad
    \theta_2 = \pm \arccos(\cos\theta_2), \qquad
    \theta_1 = \operatorname{atan2}(y, x) - \operatorname{atan2}(l_2 \sin\theta_2,\; l_1 + l_2 \cos\theta_2).$$

    The sign of $\theta_2$ gives **two solutions**, elbow down and elbow up.
    They become one on the border of the workspace, an annulus of radii
    $|l_1 - l_2|$ and $l_1 + l_2$, and there is none outside it. **Move the
    target** around and out of reach.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    l1 = mo.ui.slider(start=0.2, stop=1.0, step=0.05, value=0.6, label="l1 (m)", show_value=True)
    l2 = mo.ui.slider(start=0.2, stop=1.0, step=0.05, value=0.4, label="l2 (m)", show_value=True)
    tx = mo.ui.slider(start=-1.5, stop=1.5, step=0.01, value=0.55, label="target x (m)", show_value=True)
    ty = mo.ui.slider(start=-1.5, stop=1.5, step=0.01, value=0.45, label="target y (m)", show_value=True)
    mo.vstack([mo.hstack([l1, l2], justify="start", gap=2), mo.hstack([tx, ty], justify="start", gap=2)])
    return l1, l2, tx, ty


@app.cell
def _(np):
    def ik_planar_2r(x, y, l1, l2):
        """Joint angles (theta1, theta2) of a planar 2R arm reaching (x, y): elbow down, elbow up, or none."""
        c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        if abs(c2) > 1:
            return []
        solutions = []
        for s2 in (np.sqrt(1 - c2 * c2), -np.sqrt(1 - c2 * c2)):
            theta2 = np.arctan2(s2, c2)
            theta1 = np.arctan2(y, x) - np.arctan2(l2 * s2, l1 + l2 * c2)
            solutions.append(np.array([theta1, theta2]))
        return solutions
    return (ik_planar_2r,)


@app.cell
def _(forward_kinematics, ik_planar_2r, l1, l2, tx, ty):
    planar_links = [("R", 0, 0, l1.value, 0), ("R", 0, 0, l2.value, 0)]
    planar_solutions = ik_planar_2r(tx.value, ty.value, l1.value, l2.value)
    # Check each solution with the forward kinematics:
    planar_poses = [forward_kinematics(planar_links, s) for s in planar_solutions]
    return planar_poses, planar_solutions


@app.cell(hide_code=True)
def _(go, l1, l2, mo, np, planar_poses, planar_solutions, tx, ty):
    _fig = go.Figure()
    _t = np.linspace(0, 2 * np.pi, 200)
    _rin, _rout = abs(l1.value - l2.value), l1.value + l2.value
    _fig.add_trace(
        go.Scatter(
            x=np.r_[_rout * np.cos(_t), _rin * np.cos(_t[::-1])], y=np.r_[_rout * np.sin(_t), _rin * np.sin(_t[::-1])],
            fill="toself", fillcolor="rgba(31,119,180,0.12)", line=dict(width=0), name="workspace", hoverinfo="skip",
        )
    )
    _names = ["elbow down (θ2 > 0)", "elbow up (θ2 < 0)"]
    for _k, (_poses, _color) in enumerate(zip(planar_poses, ("#d62728", "#2ca02c"))):
        _fig.add_trace(
            go.Scatter(
                x=[p.x for p in _poses], y=[p.y for p in _poses], mode="lines+markers", name=_names[_k],
                line=dict(color=_color, width=5), marker=dict(size=9, color="#ffbf00", line=dict(color="black", width=1)),
            )
        )
    _fig.add_trace(
        go.Scatter(
            x=[tx.value], y=[ty.value], mode="markers", name="target",
            marker=dict(symbol="x", size=14, color="black"),
        )
    )
    _fig.update_layout(
        height=480, margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(range=[-1.6, 1.6], title="x (m)"),
        yaxis=dict(range=[-1.6, 1.6], title="y (m)", scaleanchor="x"), legend=dict(x=0, y=1),
    )

    if planar_solutions:
        _rows = [
            f"| {_names[k]} | {np.degrees(s[0]):.1f} | {np.degrees(s[1]):.1f} | "
            f"{1000 * np.hypot(p[-1].x - tx.value, p[-1].y - ty.value):.1e} |"
            for k, (s, p) in enumerate(zip(planar_solutions, planar_poses))
        ]
        _text = mo.md(
            "| Solution | θ1 (deg) | θ2 (deg) | FK error (mm) |\n|---|---|---|---|\n" + "\n".join(_rows)
            + ("\n\nOn the border of the workspace: both solutions are the same." if abs(planar_solutions[0][1]) < 1e-3 or abs(abs(planar_solutions[0][1]) - np.pi) < 1e-3 else "")
        )
    else:
        _text = mo.callout(
            mo.md(
                f"**Out of reach**: the target is {np.hypot(tx.value, ty.value):.2f} m from the base, outside "
                f"the workspace ({_rin:.2f} m to {_rout:.2f} m), so $|\\cos\\theta_2| > 1$."
            ),
            kind="warn",
        )
    mo.hstack([_fig, _text], widths=[3, 2], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Kinematic decoupling: a 6-DOF arm with a spherical wrist

    Most industrial arms have a **spherical wrist**: the axes of the last
    three joints meet at one point, the **wrist center** $p_w$. Then the
    problem splits into two smaller ones (Pieper's method). The
    anthropomorphic arm used here has this DH table (lengths in meters):

    | Link | $\theta_i$ | $d_i$ | $a_i$ | $\alpha_i$ |
    |---|---|---|---|---|
    | 1 | $q_1$ | 0.4 | 0 | 90° |
    | 2 | $q_2$ | 0 | 0.5 | 0 |
    | 3 | $q_3$ | 0 | 0 | 90° |
    | 4 | $q_4$ | 0.5 | 0 | -90° |
    | 5 | $q_5$ | 0 | 0 | 90° |
    | 6 | $q_6$ | 0.1 | 0 | 0 |

    1. **Position.** The tool lies $d_6$ along its approach axis $z_6$ from
       the wrist center, so $p_w = p_e - d_6\, z_6$, and $p_w$ only depends on
       $\theta_1, \theta_2, \theta_3$. Joint 1 turns the arm towards it:
       $\theta_1 = \operatorname{atan2}(p_{w,y}, p_{w,x})$, or that plus 180°
       (reaching backwards). In the vertical plane of the arm, the upper arm
       ($a_2$) and the forearm ($d_4$) are a planar 2R arm, solved as in
       section 1, with $\theta_3$ shifted 90° because the forearm lies along
       $z_3$.
    2. **Orientation.** With $\theta_{1,2,3}$ known, so is the rotation
       $R_0^3$ of frame 3, and the wrist must provide the rest:
       $R_3^6 = (R_0^3)^\top R_e$. The spherical wrist is a set of $ZYZ$ Euler
       angles:

    $$R_3^6 = \begin{bmatrix}
    c_4 c_5 c_6 - s_4 s_6 & -c_4 c_5 s_6 - s_4 c_6 & c_4 s_5 \\
    s_4 c_5 c_6 + c_4 s_6 & -s_4 c_5 s_6 + c_4 c_6 & s_4 s_5 \\
    -s_5 c_6 & s_5 s_6 & c_5
    \end{bmatrix}
    \;\Rightarrow\;
    \begin{aligned}
    \theta_5 &= \operatorname{atan2}\big(\pm\sqrt{r_{13}^2 + r_{23}^2},\, r_{33}\big) \\
    \theta_4 &= \operatorname{atan2}(\pm r_{23}, \pm r_{13}) \\
    \theta_6 &= \operatorname{atan2}(\pm r_{32}, \mp r_{31})
    \end{aligned}$$

    Two choices for the shoulder, two for the elbow and two for the wrist:
    **up to eight solutions** for the same tool pose. **Move the target** and
    pick a solution to highlight. Near $\theta_5 = 0$ (a wrist singularity)
    only $\theta_4 + \theta_6$ is defined.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    px = mo.ui.slider(start=-1.0, stop=1.0, step=0.01, value=0.6, label="x (m)", show_value=True)
    py = mo.ui.slider(start=-1.0, stop=1.0, step=0.01, value=0.2, label="y (m)", show_value=True)
    pz = mo.ui.slider(start=-0.4, stop=1.4, step=0.01, value=0.7, label="z (m)", show_value=True)
    pyaw = mo.ui.slider(start=-180, stop=180, step=1, value=0, label="yaw (deg)", show_value=True)
    ppitch = mo.ui.slider(start=-90, stop=90, step=1, value=20, label="pitch (deg)", show_value=True)
    proll = mo.ui.slider(start=-180, stop=180, step=1, value=180, label="roll (deg)", show_value=True)
    solution_k = mo.ui.slider(start=1, stop=8, step=1, value=3, label="Highlighted solution", show_value=True)
    mo.vstack(
        [
            mo.md("### Target tool pose"),
            mo.hstack([px, py, pz], justify="start", gap=2),
            mo.hstack([pyaw, ppitch, proll], justify="start", gap=2),
            solution_k,
        ]
    )
    return ppitch, proll, px, py, pyaw, pz, solution_k


@app.cell
def _(forward_kinematics, ik_planar_2r, np, origin):
    ANTHROPOMORPHIC = [
        ("R", 0, 0.4, 0, 90), ("R", 0, 0, 0.5, 0), ("R", 0, 0, 0, 90),
        ("R", 0, 0.5, 0, -90), ("R", 0, 0, 0, 90), ("R", 0, 0.1, 0, 0),
    ]

    def ik_anthropomorphic(target, d1=0.4, a2=0.5, d4=0.5, d6=0.1):
        """All the closed-form solutions (radians) and their labels, for the arm above."""
        R_e = np.asarray(target.getRotationMatrix())
        p_w = origin(target) - d6 * R_e[:, 2]  # the wrist center
        solutions, labels = [], []
        for shoulder, sign in (("front", 1), ("back", -1)):
            theta1 = np.arctan2(sign * p_w[1], sign * p_w[0])
            # In the vertical plane of the arm: horizontal reach X and height Y over the shoulder.
            X, Y = sign * np.hypot(p_w[0], p_w[1]), p_w[2] - d1
            for elbow, (q2, q3) in zip(("down", "up"), ik_planar_2r(X, Y, a2, d4)):
                theta2, theta3 = q2, q3 + np.pi / 2
                R_03 = np.asarray(forward_kinematics(ANTHROPOMORPHIC[:3], [theta1, theta2, theta3])[-1].getRotationMatrix())
                r = R_03.T @ R_e  # R_3^6, the rotation left to the wrist
                for wrist, s in (("no flip", 1), ("flip", -1)):
                    theta5 = np.arctan2(s * np.hypot(r[0, 2], r[1, 2]), r[2, 2])
                    theta4 = np.arctan2(s * r[1, 2], s * r[0, 2])
                    theta6 = np.arctan2(s * r[2, 1], -s * r[2, 0])
                    q = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
                    solutions.append(np.arctan2(np.sin(q), np.cos(q)))  # angles in (-180, 180] deg
                    labels.append(f"{shoulder}, elbow {elbow}, {wrist}")
        return solutions, labels, p_w
    return ANTHROPOMORPHIC, ik_anthropomorphic


@app.cell
def _(
    ANTHROPOMORPHIC,
    CPose3D,
    ppitch,
    proll,
    px,
    py,
    pyaw,
    pz,
    forward_kinematics,
    ik_anthropomorphic,
    np,
    pose_error,
):
    arm_target = CPose3D(
        px.value, py.value, pz.value, np.radians(pyaw.value), np.radians(ppitch.value), np.radians(proll.value)
    )
    arm_solutions, arm_labels, wrist_center = ik_anthropomorphic(arm_target)
    arm_poses = [forward_kinematics(ANTHROPOMORPHIC, s) for s in arm_solutions]
    # Check: the forward kinematics of every solution must give back the target pose.
    arm_errors = [pose_error(arm_target, p[-1]) for p in arm_poses]
    return arm_errors, arm_labels, arm_poses, arm_solutions, arm_target, wrist_center


@app.cell(hide_code=True)
def _(
    ANTHROPOMORPHIC,
    arm_errors,
    arm_labels,
    arm_poses,
    arm_solutions,
    arm_target,
    arm_traces,
    frame_traces,
    go,
    mo,
    np,
    scene_figure,
    solution_k,
    wrist_center,
):
    if not arm_solutions:
        _out = mo.callout(
            mo.md(
                f"**Out of reach**: the wrist center would be at "
                f"({wrist_center[0]:.2f}, {wrist_center[1]:.2f}, {wrist_center[2]:.2f}), "
                f"{np.linalg.norm(wrist_center - [0, 0, 0.4]):.2f} m from the shoulder, but the arm reaches 1.0 m."
            ),
            kind="warn",
        )
    else:
        _k = min(solution_k.value, len(arm_solutions)) - 1
        _rows = [
            f"| {'**' if i == _k else ''}{i + 1}{'**' if i == _k else ''} | {lab} | "
            + " | ".join(f"{v:.0f}" for v in np.degrees(s))
            + f" | {1000 * np.linalg.norm(e[:3]):.0e} | {np.degrees(np.linalg.norm(e[3:])):.0e} |"
            for i, (s, lab, e) in enumerate(zip(arm_solutions, arm_labels, arm_errors))
        ]
        _traces = []
        for i, (_p, _s) in enumerate(zip(arm_poses, arm_solutions)):
            if i != _k:
                _traces += arm_traces(ANTHROPOMORPHIC, _p, _s, color="#b0b0b0", width=5)
        _traces += arm_traces(ANTHROPOMORPHIC, arm_poses[_k], arm_solutions[_k], color="#d62728", name=f"solution {_k + 1}")
        _traces += frame_traces(arm_poses[_k][1:-1], length=0.08)
        _traces += frame_traces([arm_target], length=0.2, width=8)
        _traces.append(
            go.Scatter3d(
                x=[wrist_center[0]], y=[wrist_center[1]], z=[wrist_center[2]], mode="markers+text",
                text=["wrist center"], textposition="top center", marker=dict(size=6, color="black"), showlegend=False,
            )
        )
        _out = mo.vstack(
            [
                scene_figure(_traces, f"{len(arm_solutions)} solutions (solution {_k + 1} in red)"),
                mo.md(
                    "| # | Configuration | θ1 | θ2 | θ3 | θ4 | θ5 | θ6 | FK position error (mm) | FK rotation error (deg) |\n"
                    "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows)
                    + "\n\nAngles in degrees. The errors come from composing the six `CPose3D` links of each solution. "
                    "Only two arm shapes appear: the elbow must be at one of the two points that are $a_2$ from the "
                    "shoulder and $d_4$ from the wrist center. A \"back\" solution puts the links at the same places "
                    "as a \"front\" one, but upside down: compare the frames of the joints."
                ),
            ]
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Numerical inverse kinematics

    Closed forms only exist for some geometries. The general method is
    iterative: start at a guess $q_0$, measure the **pose error**
    $e(q) = \begin{bmatrix} p_d - p(q) \\ \log\big(R_d\, R(q)^\top\big) \end{bmatrix}$
    (position, and the rotation vector of the remaining rotation) and move the
    joints to reduce it, using the Jacobian $J(q)$, which relates small joint
    motions to small tool motions:

    | Method | Update $\Delta q$ | Notes |
    |---|---|---|
    | Jacobian transpose | $\alpha\, J^\top e$ | gradient descent: cheap and stable, but slow |
    | Pseudo-inverse (Newton-Raphson) | $J^{+} e$ | very fast near the solution; huge steps near singularities |
    | Damped least squares (Levenberg-Marquardt) | $J^\top (J J^\top + \lambda^2 I)^{-1} e$ | a compromise; $\lambda$ limits the steps |

    Here it is applied to the **UR5**, whose wrist is not spherical. The
    initial guess decides which of its solutions is found. Things to try:

    - Change the initial guess and compare the final configurations.
    - Move the target **out of reach** (e.g. $x = -1$ m): the pseudo-inverse
      jumps around (the tool moves more than 1 m in some iterations), while
      damped least squares takes short steps and, with $\lambda \approx 0.2$,
      settles at the closest pose.
    - Set $\lambda = 0$: damped least squares becomes the pseudo-inverse.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ux = mo.ui.slider(start=-1.0, stop=1.0, step=0.01, value=-0.4, label="x (m)", show_value=True)
    uy = mo.ui.slider(start=-1.0, stop=1.0, step=0.01, value=-0.35, label="y (m)", show_value=True)
    uz = mo.ui.slider(start=-0.5, stop=1.0, step=0.01, value=0.45, label="z (m)", show_value=True)
    uyaw = mo.ui.slider(start=-180, stop=180, step=1, value=60, label="yaw (deg)", show_value=True)
    upitch = mo.ui.slider(start=-90, stop=90, step=1, value=0, label="pitch (deg)", show_value=True)
    uroll = mo.ui.slider(start=-180, stop=180, step=1, value=180, label="roll (deg)", show_value=True)
    damping = mo.ui.slider(start=0, stop=0.5, step=0.01, value=0.05, label="λ, damping", show_value=True)
    guess = mo.ui.dropdown(
        options=["home", "random 1", "random 2", "random 3", "random 4", "random 5"], value="home",
        label="Initial guess",
    )
    shown = mo.ui.dropdown(
        options=["Jacobian transpose", "Pseudo-inverse", "Damped least squares"], value="Damped least squares",
        label="Arm shown",
    )
    mo.vstack(
        [
            mo.md("### Target tool pose of the UR5"),
            mo.hstack([ux, uy, uz], justify="start", gap=2),
            mo.hstack([uyaw, upitch, uroll], justify="start", gap=2),
            mo.hstack([guess, damping, shown], justify="start", gap=2),
        ]
    )
    return damping, guess, shown, upitch, uroll, ux, uy, uyaw, uz


@app.cell
def _(forward_kinematics, np, origin, pose_error):
    UR5 = [
        ("R", 0, 0.089159, 0, 90), ("R", 0, 0, -0.425, 0), ("R", 0, 0, -0.39225, 0),
        ("R", 0, 0.10915, 0, 90), ("R", 0, 0.09465, 0, -90), ("R", 0, 0.0823, 0, 0),
    ]
    UR5_HOME = np.radians([0, -60, 90, -120, -90, 0])

    def jacobian(links, poses):
        """Geometric Jacobian (6 x n) of an arm with revolute joints."""
        o_n = origin(poses[-1])
        J = np.zeros((6, len(links)))
        for i in range(len(links)):
            z = np.asarray(poses[i].getRotationMatrix())[:, 2]
            J[:3, i] = np.cross(z, o_n - origin(poses[i]))
            J[3:, i] = z
        return J

    def ik_numerical(links, target, q0, method, damping=0.05, max_iters=150, tol=1e-6):
        """Iterates from q0; returns the final q, the error norm and the tool position at every iteration."""
        q = np.array(q0, dtype=float)
        errors, path = [], []
        for _ in range(max_iters):
            poses = forward_kinematics(links, q)
            e = pose_error(target, poses[-1])
            errors.append(np.linalg.norm(e))
            path.append(origin(poses[-1]))
            if errors[-1] < tol:
                break
            J = jacobian(links, poses)
            if method == "Jacobian transpose":
                JJte = J @ J.T @ e
                step = J.T @ e * (e @ JJte) / max(JJte @ JJte, 1e-12)  # the best step length along J^T e
            elif method == "Pseudo-inverse":
                step = np.linalg.pinv(J) @ e
            else:
                step = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            q = q + step
        q = np.arctan2(np.sin(q), np.cos(q))  # angles in (-180, 180] deg
        return q, np.array(errors), np.array(path)
    return UR5, UR5_HOME, ik_numerical


@app.cell
def _(
    CPose3D,
    UR5,
    UR5_HOME,
    damping,
    guess,
    ik_numerical,
    np,
    upitch,
    uroll,
    ux,
    uy,
    uyaw,
    uz,
):
    ur5_target = CPose3D(ux.value, uy.value, uz.value, np.radians(uyaw.value), np.radians(upitch.value), np.radians(uroll.value))
    if guess.value == "home":
        ur5_q0 = UR5_HOME
    else:
        ur5_q0 = np.random.default_rng(int(guess.value.split()[-1])).uniform(-np.pi, np.pi, 6)
    ur5_results = {
        m: ik_numerical(UR5, ur5_target, ur5_q0, m, damping=damping.value)
        for m in ("Jacobian transpose", "Pseudo-inverse", "Damped least squares")
    }
    return ur5_q0, ur5_results, ur5_target


@app.cell(hide_code=True)
def _(
    UR5,
    arm_traces,
    forward_kinematics,
    frame_traces,
    go,
    mo,
    np,
    scene_figure,
    shown,
    ur5_q0,
    ur5_results,
    ur5_target,
):
    _colors = {"Jacobian transpose": "#9467bd", "Pseudo-inverse": "#ff7f0e", "Damped least squares": "#2ca02c"}
    _conv = go.Figure()
    for _m, (_q, _err, _path) in ur5_results.items():
        _conv.add_trace(go.Scatter(y=_err, mode="lines", name=_m, line=dict(color=_colors[_m], width=3)))
    _conv.update_layout(
        height=420, margin=dict(l=0, r=0, t=40, b=0), title="Convergence", legend=dict(x=0.4, y=1),
        xaxis_title="iteration", yaxis=dict(type="log", title="pose error |e|", exponentformat="power"),
    )

    _q, _err, _path = ur5_results[shown.value]
    _traces = arm_traces(UR5, forward_kinematics(UR5, ur5_q0), ur5_q0, color="#b0b0b0", name="initial guess")
    _traces += arm_traces(UR5, forward_kinematics(UR5, _q), _q, color=_colors[shown.value], name=shown.value)
    _traces.append(
        go.Scatter3d(
            x=_path[:, 0], y=_path[:, 1], z=_path[:, 2], mode="lines+markers", name="tool path",
            line=dict(color="black", width=2, dash="dot"), marker=dict(size=2, color="black"),
        )
    )
    _traces += frame_traces([ur5_target], length=0.12, width=8)

    _rows = [
        f"| {m} | {len(err)} | {err[-1]:.1e} | " + ", ".join(f"{v:.0f}" for v in np.degrees(q)) + " |"
        for m, (q, err, _) in ur5_results.items()
    ]
    mo.vstack(
        [
            mo.hstack(
                [_conv, scene_figure(_traces, f"{shown.value}: final configuration", height=420)],
                widths=[2, 3],
            ),
            mo.md(
                "| Method | Iterations | Final error | Final joints (deg) |\n|---|---|---|---|\n" + "\n".join(_rows)
                + "\n\nAn error that stops decreasing means a target out of reach, or a method stuck near a singularity."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
