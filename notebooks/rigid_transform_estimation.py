# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
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
    # Rigid transformations from point correspondences

    Given pairs of corresponding points measured in two frames, which rigid
    transformation (rotation plus translation) best maps one set onto the
    other? This problem is at the core of landmark-based localization,
    point-cloud registration and camera pose estimation. This notebook uses
    the [MRPT](https://www.mrpt.org/) Python wheel (`pip install mrpt3`) to
    solve it in 2D and 3D, with the `mrpt.tfest` module:

    - `se2_l2`: closed-form least-squares SE(2) solution.
    - `se3_l2_robust`: RANSAC around Horn's closed-form SE(3) solution.

    plus `mrpt.poses` (`CPose2D`, `CPose3D`, `CPose3DQuat`) for the poses.
    Real correspondences are noisy, and some are plain wrong (outliers).
    **Move the sliders**: a few outliers are enough to ruin least squares,
    while RANSAC ignores them, up to a point.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from mrpt.math import TPoint3Df
    from mrpt.poses import CPose2D, CPose3D, CPose3DQuat
    from mrpt.random import getRandomGenerator
    from mrpt.tfest import TMatchingPair, TMatchingPairList, TSE3RobustParams, se2_l2, se3_l2_robust
    return (
        CPose2D,
        CPose3D,
        CPose3DQuat,
        TMatchingPair,
        TMatchingPairList,
        TPoint3Df,
        TSE3RobustParams,
        getRandomGenerator,
        go,
        make_subplots,
        mo,
        np,
        plt,
        se2_l2,
        se3_l2_robust,
    )


@app.cell(hide_code=True)
def _(mo):
    n_points = mo.ui.slider(start=20, stop=150, step=5, value=60, label="Correspondences", show_value=True)
    noise = mo.ui.slider(start=0, stop=20, step=0.5, value=3, label="Noise σ (cm)", show_value=True)
    outliers = mo.ui.slider(start=0, stop=0.8, step=0.05, value=0.3, label="Outlier ratio", show_value=True)
    iterations = mo.ui.slider(start=10, stop=500, step=10, value=200, label="RANSAC iterations", show_value=True)
    threshold = mo.ui.slider(start=1, stop=50, step=1, value=10, label="RANSAC threshold (cm)", show_value=True)

    mo.vstack(
        [
            mo.md("### Parameters"),
            mo.hstack([n_points, noise, outliers], justify="start", gap=2),
            mo.hstack([iterations, threshold], justify="start", gap=2),
        ]
    )
    return iterations, n_points, noise, outliers, threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How the estimators work

    Each correspondence pairs a point $p_i$ measured in the sensor (local)
    frame with a point $g_i$ in the map (global) frame. The transformation $T$
    we look for satisfies $g_i \approx T \oplus p_i$.

    - **Least squares** minimizes $\sum_i \lVert g_i - T \oplus p_i \rVert^2$
      in closed form. Fast and optimal for Gaussian noise, but one wrong pair
      far away pulls the whole solution.
    - **RANSAC** repeatedly fits $T$ to 5 random pairs, collects the pairs
      that agree with it (adding one moves the fitted pose by less than the
      threshold, and rotates it less than 1°), and keeps the largest consensus
      set. It succeeds if at least one random 5-pair sample has no outliers.
    """)
    return


@app.cell
def _(
    TMatchingPair,
    TMatchingPairList,
    TPoint3Df,
    TSE3RobustParams,
    getRandomGenerator,
    np,
    se3_l2_robust,
):
    def make_pairs(local_pts, global_pts):
        """A TMatchingPairList from two (N, 3) arrays of corresponding points."""
        pairs = TMatchingPairList()
        for i, (p, g) in enumerate(zip(local_pts, global_pts)):
            pair = TMatchingPair()
            pair.localIdx = pair.globalIdx = i
            pair.local_pt = TPoint3Df(*p)
            pair.global_pt = TPoint3Df(*g)
            pairs.push_back(pair)
        return pairs

    def corrupt(global_pts, ratio, rng):
        """Pairs a fraction of the points with a wrong, randomly chosen, map point. Returns the outlier mask."""
        n = len(global_pts)
        idx = rng.choice(n, size=int(round(ratio * n)), replace=False)
        global_pts[idx] = global_pts[(idx + rng.integers(1, n, size=len(idx))) % n]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        return mask

    def horn_least_squares(pairs):
        """Plain least-squares SE(3) fit with all the pairs (Horn's method).

        A RANSAC run whose random sample is the whole set does exactly one such fit."""
        params = TSE3RobustParams()
        params.ransac_minSetSize = len(pairs)
        params.ransac_maxSetSizePct = 1.0
        params.ransac_nmaxSimulations = 1
        _ok, result = se3_l2_robust(pairs, params)
        return result.transformation

    def ransac(pairs, n_iters, threshold_m):
        """Robust SE(3) fit. Returns (CPose3D, inlier mask), or (None, None) if it fails."""
        params = TSE3RobustParams()
        params.ransac_minSetSize = 5  # pairs per random hypothesis
        params.ransac_nmaxSimulations = n_iters
        params.ransac_maxSetSizePct = 0.25  # smallest accepted consensus set: 25% of the pairs
        params.ransac_threshold_lin = threshold_m
        getRandomGenerator().randomize(1)  # repeatable random samples
        ok, result = se3_l2_robust(pairs, params)
        if not ok:
            return None, None
        inliers = np.zeros(len(pairs), dtype=bool)
        inliers[list(result.inliers_idx)] = True
        return result.transformation, inliers

    def transform(pose, pts):
        """Applies a CPose3D to an (N, 3) array of points."""
        H = np.asarray(pose.getHomogeneousMatrix())
        return pts @ H[:3, :3].T + H[:3, 3]
    return corrupt, horn_least_squares, make_pairs, ransac, transform


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. SE(2): robot pose from landmarks

    A robot observes landmarks whose positions in the map are known. Each
    observation, in the robot frame, is matched to a map landmark, and some
    matches are wrong. The transformation from robot to map frame is the
    robot pose $(x, y, \phi)$. For RANSAC we use `se3_l2_robust` with all
    $z = 0$: a planar problem is a special case of SE(3).
    """)
    return


@app.cell
def _(
    CPose2D,
    CPose3D,
    corrupt,
    iterations,
    make_pairs,
    n_points,
    noise,
    np,
    outliers,
    ransac,
    se2_l2,
    threshold,
    transform,
):
    rng2 = np.random.default_rng(2)
    true_pose2 = CPose2D(2.0, 1.0, np.radians(35.0))

    landmarks = np.column_stack([rng2.uniform(-8.0, 8.0, (n_points.value, 2)), np.zeros(n_points.value)])
    # Observations: landmarks in the robot frame, plus sensor noise in (x, y):
    observed2 = transform(CPose3D() - CPose3D(true_pose2), landmarks)
    observed2[:, :2] += rng2.normal(0.0, noise.value / 100.0, (n_points.value, 2))
    matched2 = landmarks.copy()
    is_outlier2 = corrupt(matched2, outliers.value, rng2)

    pairs2 = make_pairs(observed2, matched2)
    _ok, _est = se2_l2(pairs2)  # returns a TPose2D
    ls_pose2 = CPose2D.fromTPose(_est)
    _robust, inliers2 = ransac(pairs2, iterations.value, threshold.value / 100.0)
    ransac_pose2 = CPose2D(_robust) if _robust is not None else None
    return (
        inliers2,
        is_outlier2,
        landmarks,
        ls_pose2,
        matched2,
        observed2,
        ransac_pose2,
        true_pose2,
    )


@app.cell
def _(CPose3DQuat, np):
    def pose_error(estimate, truth):
        """Translation (cm) and rotation (deg) errors of a CPose3D estimate."""
        if estimate is None:
            return np.nan, np.nan
        err = estimate - truth  # the estimate, seen from the true pose
        angle = 2.0 * np.arccos(min(1.0, abs(CPose3DQuat(err).quat.r)))
        return 100.0 * np.linalg.norm([err.x, err.y, err.z]), np.degrees(angle)

    def results_table(rows, n, true_outliers):
        """Markdown table comparing estimators. rows: (name, pose or None, errors, inlier mask or None)."""
        lines = [
            "| Method | Translation error (cm) | Rotation error (deg) | Pairs used | Outliers among them |",
            "|---|---|---|---|---|",
        ]
        for name, pose, err, used in rows:
            if pose is None:
                lines.append(f"| {name} | failed | failed | - | - |")
                continue
            used = np.ones(n, dtype=bool) if used is None else used
            lines.append(
                f"| {name} | {err[0]:.1f} | {err[1]:.2f} | {used.sum()} | {np.sum(used & true_outliers)} |"
            )
        return "\n".join(lines)

    def success_chance(outlier_ratio, n_iters, sample_size=5):
        """Probability that at least one RANSAC sample is free of outliers."""
        return 1.0 - (1.0 - (1.0 - outlier_ratio) ** sample_size) ** n_iters
    return pose_error, results_table, success_chance


@app.cell(hide_code=True)
def _(
    CPose3D,
    inliers2,
    is_outlier2,
    iterations,
    ls_pose2,
    mo,
    n_points,
    outliers,
    pose_error,
    ransac_pose2,
    results_table,
    success_chance,
    true_pose2,
):
    _truth = CPose3D(true_pose2)
    _rows = [
        ("Least squares (`se2_l2`)", ls_pose2, pose_error(CPose3D(ls_pose2), _truth), None),
        (
            "RANSAC (`se3_l2_robust`)",
            ransac_pose2,
            pose_error(CPose3D(ransac_pose2) if ransac_pose2 is not None else None, _truth),
            inliers2,
        ),
    ]
    mo.md(
        results_table(_rows, n_points.value, is_outlier2)
        + f"\n\nTrue outliers: **{is_outlier2.sum()}** of {n_points.value}. "
        f"Chance that one of the {iterations.value} RANSAC samples is outlier-free: "
        f"**{100 * success_chance(outliers.value, iterations.value):.1f}%**."
    )
    return


@app.cell(hide_code=True)
def _(
    CPose3D,
    inliers2,
    is_outlier2,
    landmarks,
    ls_pose2,
    matched2,
    np,
    observed2,
    plt,
    ransac_pose2,
    transform,
    true_pose2,
):
    def draw_pose(ax, pose, color, label, ls="-"):
        ax.annotate(
            "", xy=(pose.x + 1.5 * np.cos(pose.phi), pose.y + 1.5 * np.sin(pose.phi)), xytext=(pose.x, pose.y),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.5, ls=ls),
        )
        ax.plot(pose.x, pose.y, "o", color=color, mfc="none" if ls != "-" else color, mew=2, ms=8, label=label)

    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    for ax, pose, kept, title in [
        (axs2[0], ls_pose2, np.ones(len(observed2), dtype=bool), "Least squares (se2_l2)"),
        (axs2[1], ransac_pose2, inliers2, "RANSAC (se3_l2_robust, z = 0)"),
    ]:
        ax.scatter(*landmarks[:, :2].T, s=60, facecolors="none", edgecolors="0.55", label="Map landmarks")
        if pose is None:
            ax.text(
                0.5, 0.97, "RANSAC failed", transform=ax.transAxes, ha="center", va="top", fontsize=13,
                color="tab:red", bbox=dict(facecolor="white", edgecolor="tab:red"),
            )
        else:
            projected = transform(CPose3D(pose), observed2)
            ax.scatter(*projected[kept, :2].T, s=12, color="tab:blue", label="Observations, placed by the estimate")
            if not kept.all():
                ax.scatter(*projected[~kept, :2].T, s=30, marker="x", color="tab:red", label="Rejected by RANSAC")
            draw_pose(ax, pose, "tab:blue", "Estimated robot pose")
        # Wrong matches: from the true landmark position to the wrongly matched one.
        true_pos = landmarks[is_outlier2, :2]
        for (x0, y0), (x1, y1) in zip(true_pos, matched2[is_outlier2, :2]):
            ax.plot([x0, x1], [y0, y1], color="tab:red", lw=0.6, alpha=0.5)
        ax.plot([], [], color="tab:red", lw=0.6, label="Wrong matches (ground truth)")
        draw_pose(ax, true_pose2, "black", "True robot pose", ls="--")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (m)")
    axs2[0].set_ylabel("y (m)")
    _legend = {}  # label -> handle, without duplicates
    for ax in axs2:
        _legend.update(zip(*reversed(ax.get_legend_handles_labels())))
    fig2.legend(_legend.values(), _legend.keys(), loc="lower center", ncol=3, fontsize=8)
    fig2.tight_layout(rect=(0, 0.08, 1, 1))
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. SE(3): pose of a 3D sensor

    A 3D sensor sees points on the faces of a box-shaped object whose model
    is known, and each sensed point is matched to a model point. Now the
    transformation has six degrees of freedom. Drag the 3D views to rotate
    them: with outliers, the least-squares fit leaves the sensed points
    (colored) visibly off the model (gray).
    """)
    return


@app.cell
def _(
    CPose3D,
    corrupt,
    horn_least_squares,
    iterations,
    make_pairs,
    n_points,
    noise,
    np,
    outliers,
    ransac,
    threshold,
    transform,
):
    rng3 = np.random.default_rng(3)
    true_pose3 = CPose3D(1.0, -0.5, 0.4, np.radians(30.0), np.radians(10.0), np.radians(-15.0))

    # Model: points on the faces of a 4 x 3 x 2 m box.
    BOX_SIZE = np.array([4.0, 3.0, 2.0])
    model = rng3.uniform(0.0, 1.0, (n_points.value, 3)) * BOX_SIZE
    _face = rng3.integers(0, 3, n_points.value)
    model[np.arange(n_points.value), _face] = rng3.integers(0, 2, n_points.value) * BOX_SIZE[_face]

    sensed = transform(CPose3D() - true_pose3, model)  # in the sensor frame
    sensed += rng3.normal(0.0, noise.value / 100.0, sensed.shape)
    matched3 = model.copy()
    is_outlier3 = corrupt(matched3, outliers.value, rng3)

    pairs3 = make_pairs(sensed, matched3)
    ls_pose3 = horn_least_squares(pairs3)
    ransac_pose3, inliers3 = ransac(pairs3, iterations.value, threshold.value / 100.0)
    return BOX_SIZE, inliers3, is_outlier3, ls_pose3, model, ransac_pose3, sensed, true_pose3


@app.cell(hide_code=True)
def _(
    inliers3,
    is_outlier3,
    ls_pose3,
    mo,
    n_points,
    pose_error,
    ransac_pose3,
    results_table,
    true_pose3,
):
    _rows = [
        ("Least squares (Horn)", ls_pose3, pose_error(ls_pose3, true_pose3), None),
        ("RANSAC (`se3_l2_robust`)", ransac_pose3, pose_error(ransac_pose3, true_pose3), inliers3),
    ]
    mo.md(results_table(_rows, n_points.value, is_outlier3))
    return


@app.cell(hide_code=True)
def _(BOX_SIZE, go, inliers3, ls_pose3, make_subplots, model, np, ransac_pose3, sensed, transform):
    def points_trace(pts, color, name, symbol="circle", size=3, show=True):
        return go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=name, showlegend=show,
            marker=dict(size=size, color=color, symbol=symbol),
        )

    def box_trace(size):
        """The 12 edges of the model box, as one line trace."""
        corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * size
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
        xyz = np.array([p for a, b in edges for p in (corners[a], corners[b], [None] * 3)], dtype=object)
        return go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", line=dict(color="#bbbbbb", width=2),
            hoverinfo="skip", showlegend=False,
        )

    fig3 = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Least squares (Horn)", "RANSAC (se3_l2_robust)"),
        horizontal_spacing=0.02,
    )
    for _col in (1, 2):
        fig3.add_trace(box_trace(BOX_SIZE), row=1, col=_col)
    fig3.add_trace(points_trace(model, "#999999", "Model points"), row=1, col=1)
    fig3.add_trace(points_trace(transform(ls_pose3, sensed), "#1f77b4", "Sensed points, placed by the estimate"), 1, 1)
    fig3.add_trace(points_trace(model, "#999999", "Model points", show=False), row=1, col=2)
    if ransac_pose3 is not None:
        _placed = transform(ransac_pose3, sensed)
        fig3.add_trace(points_trace(_placed[inliers3], "#1f77b4", "", show=False), row=1, col=2)
        fig3.add_trace(points_trace(_placed[~inliers3], "#d62728", "Rejected by RANSAC", "x", 4), row=1, col=2)
    _scene = dict(
        aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
        camera=dict(eye=dict(x=1.4, y=-1.6, z=1.1)),
    )
    fig3.update_layout(
        height=520, margin=dict(l=0, r=0, t=40, b=0), scene=_scene, scene2=_scene,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
    )
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Things to try**

    - Set the outlier ratio to 0: both methods agree, and their error grows
      with the noise.
    - Raise the outlier ratio: least squares degrades at once, RANSAC keeps
      working until the outlier-free chance (above) drops, or until fewer than
      25% of the pairs are inliers, the smallest consensus set accepted here.
    - With large noise, a small threshold makes RANSAC reject good pairs too.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/rigid_transform_estimation.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
