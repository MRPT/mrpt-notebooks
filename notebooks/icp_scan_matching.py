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
    # 2D scan matching with ICP

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to align two simulated 2D laser scans with the
    Iterative Closest Point (ICP) algorithm. It uses three MRPT modules:

    - `mrpt.poses`: SE(2) poses (`CPose2D`) and their Gaussian uncertainty (`CPosePDFGaussian`).
    - `mrpt.maps`: point clouds (`CSimplePointsMap`), filled from NumPy arrays.
    - `mrpt.slam`: the ICP algorithm (`CICP`).

    A robot takes a 360° scan, moves, and takes a second one. ICP estimates
    that motion from the two scans alone, starting from a zero initial guess
    (as if there were no odometry). **Move the sliders**: everything below is
    recomputed. Try large rotations, or a small matching distance, to see when
    ICP gets stuck in a wrong solution.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mrpt.maps import CSimplePointsMap
    from mrpt.poses import CPose2D, CPosePDFGaussian
    from mrpt.slam import CICP, CICPOptions, TICPAlgorithm
    return (
        CICP,
        CICPOptions,
        CPose2D,
        CPosePDFGaussian,
        CSimplePointsMap,
        TICPAlgorithm,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    dx = mo.ui.slider(start=-1.5, stop=1.5, step=0.05, value=0.6, label="Robot motion: x (m)", show_value=True)
    dy = mo.ui.slider(start=-1.5, stop=1.5, step=0.05, value=0.3, label="y (m)", show_value=True)
    dyaw = mo.ui.slider(start=-90, stop=90, step=1, value=20, label="yaw (deg)", show_value=True)
    noise = mo.ui.slider(start=0.0, stop=10.0, step=0.5, value=2.0, label="Range noise σ (cm)", show_value=True)
    algorithm = mo.ui.dropdown(options=["Classic", "Levenberg-Marquardt"], value="Classic", label="ICP algorithm")
    max_dist = mo.ui.slider(start=0.1, stop=3.0, step=0.05, value=0.75, label="Initial matching distance (m)", show_value=True)
    max_iters = mo.ui.slider(start=5, stop=200, step=5, value=80, label="Max. iterations", show_value=True)

    mo.vstack(
        [
            mo.md("### Parameters"),
            mo.hstack([dx, dy, dyaw], justify="start", gap=2),
            mo.hstack([noise, algorithm], justify="start", gap=2),
            mo.hstack([max_dist, max_iters], justify="start", gap=2),
        ]
    )
    return algorithm, dx, dy, dyaw, max_dist, max_iters, noise


@app.cell
def _(np):
    # The simulated world: wall segments [x0, y0, x1, y1], in meters.
    def box(x0, y0, x1, y1):
        return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]

    walls = np.array(
        box(-6.0, -4.0, 6.0, 4.0)  # room
        + box(-3.5, 1.0, -2.0, 2.2)  # cabinet
        + box(2.5, -2.8, 3.3, -2.0)  # column
        + [[1.0, 4.0, 4.0, 1.5], [-6.0, -1.5, -4.0, -1.5]]  # slanted wall, partition
    )

    def simulate_scan(pose, rng, sigma=0.0, n_rays=360, max_range=15.0):
        """Ray-casts a 360 deg laser scan from pose (x, y, phi) against the walls.

        Returns the points in the sensor frame, as an (N, 2) array."""
        x, y, phi = pose
        ang = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
        d = np.stack([np.cos(ang + phi), np.sin(ang + phi)], axis=1)
        e = walls[:, 2:] - walls[:, :2]
        w = walls[:, :2] - [x, y]
        # Ray-segment intersection, solving t*d = w + u*e for all pairs at once:
        den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / den
            u = (w[None, :, 0] * d[:, None, 1] - w[None, :, 1] * d[:, None, 0]) / den
        hit = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        r = np.where(hit, t, np.inf).min(axis=1)
        ok = r < max_range
        r = r[ok] + rng.normal(0.0, sigma, size=ok.sum())
        return np.stack([r * np.cos(ang[ok]), r * np.sin(ang[ok])], axis=1)
    return simulate_scan, walls


@app.cell
def _(
    CICP,
    CICPOptions,
    CPose2D,
    CPosePDFGaussian,
    CSimplePointsMap,
    TICPAlgorithm,
    algorithm,
    dx,
    dy,
    dyaw,
    max_dist,
    max_iters,
    noise,
    np,
    simulate_scan,
):
    # Ground truth: robot pose at the first scan, and its motion (unknown to ICP).
    pose_a = CPose2D(-1.5, -1.0, np.radians(20.0))
    motion = CPose2D(dx.value, dy.value, np.radians(dyaw.value))
    pose_b = pose_a + motion  # SE(2) composition

    rng = np.random.default_rng(42)
    sigma = noise.value / 100.0
    scan_a = simulate_scan((pose_a.x, pose_a.y, pose_a.phi), rng, sigma)
    scan_b = simulate_scan((pose_b.x, pose_b.y, pose_b.phi), rng, sigma)

    def to_points_map(xy):
        m = CSimplePointsMap()
        m.setPointsFromNumpy(np.column_stack([xy, np.zeros(len(xy))]))
        return m

    opts = CICPOptions()
    opts.ICP_algorithm = (
        TICPAlgorithm.icpClassic if algorithm.value == "Classic" else TICPAlgorithm.icpLevenbergMarquardt
    )
    opts.thresholdDist = max_dist.value
    opts.maxIterations = max_iters.value

    # Pose of scan B relative to scan A, starting from a zero initial guess:
    icp = CICP(opts)
    result, info = icp.AlignPDF(to_points_map(scan_a), to_points_map(scan_b), CPosePDFGaussian())
    estimate = result.getMean()
    cov = np.array(result.getCovariance())
    error = estimate - motion  # estimate relative to the ground truth
    return cov, error, estimate, info, motion, pose_a, scan_a, scan_b


@app.cell(hide_code=True)
def _(cov, error, estimate, info, mo, motion, np):
    _err_t = 100.0 * np.hypot(error.x, error.y)
    _err_r = np.degrees(abs(error.phi))
    _sd = np.sqrt(np.diag(cov))
    _ok = _err_t < 5.0 and _err_r < 1.0
    _verdict = (
        mo.callout(mo.md(f"**ICP converged**: error {_err_t:.1f} cm, {_err_r:.2f}°."), kind="success")
        if _ok
        else mo.callout(mo.md(f"**ICP failed** (wrong local minimum): error {_err_t:.1f} cm, {_err_r:.1f}°."), kind="danger")
    )
    mo.vstack(
        [
            _verdict,
            mo.md(
                f"""
    |  | x (m) | y (m) | yaw (deg) |
    |---|---|---|---|
    | True motion | {motion.x:.3f} | {motion.y:.3f} | {np.degrees(motion.phi):.2f} |
    | ICP estimate | {estimate.x:.3f} | {estimate.y:.3f} | {np.degrees(estimate.phi):.2f} |
    | ICP 1σ (its own covariance) | {_sd[0]:.4f} | {_sd[1]:.4f} | {np.degrees(_sd[2]):.3f} |

    Iterations: **{info.nIterations}** · goodness (ratio of matched points): **{info.goodness:.2f}**
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(CPose2D, estimate, motion, np, plt, pose_a, scan_a, scan_b, walls):
    def transform(xy, pose):
        c, s = np.cos(pose.phi), np.sin(pose.phi)
        return xy @ np.array([[c, s], [-s, c]]) + [pose.x, pose.y]

    def draw_robot(ax, pose, color, ls="-"):
        ax.plot(pose.x, pose.y, "o", color=color, ms=6)
        ax.annotate(
            "",
            xy=(pose.x + 0.6 * np.cos(pose.phi), pose.y + 0.6 * np.sin(pose.phi)),
            xytext=(pose.x, pose.y),
            arrowprops=dict(arrowstyle="->", color=color, lw=2, ls=ls),
        )

    # Everything is drawn in the frame of the first scan:
    inv_a = pose_a.inverse()
    walls_a = np.hstack([transform(walls[:, :2], inv_a), transform(walls[:, 2:], inv_a)])
    origin = CPose2D()  # identity pose

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    for ax, guess, title in [
        (axs[0], origin, "Before: zero initial guess"),
        (axs[1], estimate, "After ICP"),
    ]:
        for x0, y0, x1, y1 in walls_a:
            ax.plot([x0, x1], [y0, y1], color="0.8", lw=3, zorder=0)
        ax.scatter(*scan_a.T, s=6, color="tab:blue", label="Scan A (reference)")
        ax.scatter(*transform(scan_b, guess).T, s=6, color="tab:red", label="Scan B")
        draw_robot(ax, origin, "tab:blue")
        draw_robot(ax, motion, "black", ls="--")
        draw_robot(ax, guess, "tab:red")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axs[0].legend(loc="lower left", fontsize=8)
    axs[1].plot([], [], "k--", label="True pose of B")
    axs[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/icp_scan_matching.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
