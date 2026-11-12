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
    # Monte Carlo localization with a particle filter

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to localize a simulated robot on the known map of an
    office, from its odometry and 2D laser scans. It uses these MRPT modules:

    - `mrpt.maps`: the map (`COccupancyGridMap2D` inside a `CMultiMetricMap`), configured with `mrpt.config.CConfigFileMemory`.
    - `mrpt.obs`: odometry (`CActionRobotMovement2D`, `CActionCollection`) and laser scans (`CObservation2DRangeScan`, `CSensoryFrame`).
    - `mrpt.slam` and `mrpt.bayes`: the particle filter (`CMonteCarloLocalization2D`, `CParticleFilter`).

    Monte Carlo localization (MCL) represents the belief about the robot pose
    with thousands of **particles**, each one a pose hypothesis. Choose
    **global localization** (the robot could be anywhere) or **tracking**
    (the initial pose is roughly known), then **move the time slider** to watch
    the particles converge. The filter runs once per parameter set; the time
    slider only selects which step to draw.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse

    from mrpt.bayes import CParticleFilter, TParticleFilterAlgorithm
    from mrpt.config import CConfigFileMemory
    from mrpt.maps import CMultiMetricMap, TSetOfMetricMapInitializers
    from mrpt.math import TPose2D
    from mrpt.obs import CActionCollection, CActionRobotMovement2D, CObservation2DRangeScan, CSensoryFrame
    from mrpt.poses import CPose2D
    from mrpt.random import Randomize
    from mrpt.slam import CMonteCarloLocalization2D
    return (
        CActionCollection,
        CActionRobotMovement2D,
        CConfigFileMemory,
        CMonteCarloLocalization2D,
        CMultiMetricMap,
        CObservation2DRangeScan,
        CParticleFilter,
        CPose2D,
        CSensoryFrame,
        Ellipse,
        Randomize,
        TParticleFilterAlgorithm,
        TPose2D,
        TSetOfMetricMapInitializers,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    init_mode = mo.ui.radio(
        options=["Global localization", "Tracking from a known pose"], value="Global localization", label="Initial belief"
    )
    n_particles = mo.ui.slider(steps=[300, 1000, 3000, 10000, 30000, 50000], value=30000, label="Initial particles", show_value=True)
    adaptive = mo.ui.checkbox(value=True, label="Adaptive number of particles (KLD)")
    seed = mo.ui.slider(start=0, stop=20, step=1, value=0, label="Random seed", show_value=True)

    mo.vstack(
        [
            mo.md("### Parameters"),
            mo.hstack([init_mode, mo.vstack([n_particles, adaptive, seed])], justify="start", gap=3),
        ]
    )
    return adaptive, init_mode, n_particles, seed


@app.cell
def _(np):
    # The office: wall segments [x0, y0, x1, y1], in meters.
    def box(x0, y0, x1, y1):
        return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]

    def wall_with_doors(y, x0, x1, doors, width=1.0):
        """Horizontal wall at height y, with doors centered at the given x coordinates."""
        xs = [x0] + [x for c in doors for x in (c - width / 2, c + width / 2)] + [x1]
        return [[a, y, b, y] for a, b in zip(xs[::2], xs[1::2])]

    walls = np.array(
        box(0, 0, 20, 12)  # building
        + wall_with_doors(5, 0, 20, [3.0, 10.5, 17.0])  # corridor walls
        + wall_with_doors(7, 0, 20, [2.5, 8.5, 16.0])
        + [[6.5, 0, 6.5, 5], [13.5, 0, 13.5, 5], [5, 7, 5, 12], [12, 7, 12, 12]]  # room walls
        + box(1, 1, 2.2, 1.8) + box(8, 9.5, 9.5, 11) + box(17, 2.5, 18, 3.5)  # furniture
        + [[14.5, 11, 16, 9.5]]  # slanted wall
    )

    def simulate_ranges(pose, rng, sigma=0.0, n_rays=360):
        """Ray-casts a 360 deg laser scan from pose (x, y, phi) against the walls.

        Ray i points at -pi + 2 pi i / (n_rays - 1) w.r.t. the robot heading,
        MRPT's convention for a scan with an aperture of 2 pi. Returns ranges (inf: no hit)."""
        x, y, phi = pose
        ang = np.linspace(-np.pi, np.pi, n_rays)
        d = np.stack([np.cos(ang + phi), np.sin(ang + phi)], axis=1)
        e = walls[:, 2:] - walls[:, :2]
        w = walls[:, :2] - [x, y]
        # Ray-segment intersection, solving t*d = w + u*e for all pairs at once:
        den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / den
            u = (w[None, :, 0] * d[:, None, 1] - w[None, :, 1] * d[:, None, 0]) / den
        hit = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        return np.where(hit, t, np.inf).min(axis=1) + rng.normal(0.0, sigma, size=n_rays)

    # True robot path: from a room, along the corridor, into another room. A pose every 30 cm.
    _waypoints = [(1.8, 3.2), (3, 4), (3, 6), (16, 6), (16, 9.5)]
    _xy = [np.array(_waypoints[0], float)]
    for _a, _b in zip(_waypoints[:-1], _waypoints[1:]):
        _n = max(1, round(np.hypot(_b[0] - _a[0], _b[1] - _a[1]) / 0.3))
        _xy += [np.array(_a) + (np.array(_b) - np.array(_a)) * k / _n for k in range(1, _n + 1)]
    _xy = np.array(_xy)
    _heading = np.arctan2(np.diff(_xy[:, 1]), np.diff(_xy[:, 0]))
    path = np.column_stack([_xy, np.append(_heading, _heading[-1])])  # (x, y, phi) per step
    return path, simulate_ranges, walls


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The map

    MCL needs a map. Here it is drawn from the floor plan: the cells crossed
    by walls are set as occupied, the rest of the building as free. The map
    lives in a `CMultiMetricMap` created from an INI-style configuration,
    the usual way to set MRPT map options. The `likelihoodOpts` section sets
    how scans are scored against the map (a *likelihood field*): using only
    one of every 20 rays and a wide σ gives a smoother likelihood, which makes
    global localization more robust.
    """)
    return


@app.cell
def _(CConfigFileMemory, CMultiMetricMap, TSetOfMetricMapInitializers, np, walls):
    _config = CConfigFileMemory("""
    [map]
    occupancyGrid_count=1

    [map_occupancyGrid_00_creationOpts]
    min_x=-0.5
    max_x=20.5
    min_y=-0.5
    max_y=12.5
    resolution=0.1

    [map_occupancyGrid_00_likelihoodOpts]
    LF_decimation=20
    LF_stdHit=0.6
    LF_maxCorrsDistance=0.5
    """)
    _initializers = TSetOfMetricMapInitializers()
    _initializers.loadFromConfigFile(_config, "map")
    metric_map = CMultiMetricMap(_initializers)
    grid = metric_map[0]  # the COccupancyGridMap2D

    # Rasterize the floor plan: probability of each cell being free.
    _res = grid.getResolution()
    _xs = grid.getXMin() + _res * (np.arange(grid.getSizeX()) + 0.5)
    _ys = grid.getYMin() + _res * (np.arange(grid.getSizeY()) + 0.5)
    _free = np.where(((_xs > 0) & (_xs < 20))[None, :] & ((_ys > 0) & (_ys < 12))[:, None], 1.0, 0.5)
    for _x0, _y0, _x1, _y1 in walls:
        _s = np.linspace(0, 1, int(np.hypot(_x1 - _x0, _y1 - _y0) / (0.25 * _res)) + 2)
        _ix = ((_x0 + _s * (_x1 - _x0) - grid.getXMin()) / _res).astype(int)
        _iy = ((_y0 + _s * (_y1 - _y0) - grid.getYMin()) / _res).astype(int)
        _free[_iy, _ix] = 0.0
    for _iy, _ix in zip(*np.nonzero(_free != 0.5)):
        grid.setCell(int(_ix), int(_iy), float(_free[_iy, _ix]))
    return grid, metric_map


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Odometry and laser scans

    At each step the robot moves 30 cm and reads its odometry, with an error
    of 2 cm and 1° (σ). `CActionRobotMovement2D.computeFromOdometry()` turns
    each odometry increment into a probabilistic motion model, here a Gaussian
    whose uncertainty must be at least as large as the real odometry error.
    Each laser scan goes into a `CSensoryFrame`, the set of observations
    taken at that step.
    """)
    return


@app.cell
def _(
    CActionCollection,
    CActionRobotMovement2D,
    CObservation2DRangeScan,
    CPose2D,
    CSensoryFrame,
    np,
    path,
    seed,
    simulate_ranges,
):
    MAX_RANGE = 15.0  # meters
    _rng = np.random.default_rng(seed.value)

    _motion_model = CActionRobotMovement2D.TMotionModelOptions()
    _motion_model.gaussianModel.minStdXY = 0.03  # meters
    _motion_model.gaussianModel.minStdPHI = np.radians(1.5)

    def odometry_action(increment):
        """Builds the action of moving by a (noisy) odometry increment."""
        movement = CActionRobotMovement2D()
        movement.computeFromOdometry(increment, _motion_model)
        actions = CActionCollection()
        actions.insert(movement)
        return actions

    def scan_frame(scan_ranges):
        """Wraps an array of ranges into a sensory frame with one laser scan."""
        scan = CObservation2DRangeScan()
        scan.aperture = 2 * np.pi
        scan.maxRange = MAX_RANGE
        scan.resizeScan(len(scan_ranges))
        for i, r in enumerate(scan_ranges):
            valid = bool(r < MAX_RANGE)
            scan.setScanRange(i, float(r) if valid else MAX_RANGE)
            scan.setScanRangeValidity(i, valid)
        frame = CSensoryFrame()
        frame.insert(scan)
        return frame

    gt_poses = [CPose2D(*p) for p in path]
    ranges = [simulate_ranges(p, _rng, sigma=0.02) for p in path]
    odometry_and_scans = []  # (action, sensory frame) for steps 1, 2, ...
    for _k in range(1, len(path)):
        _odometry = (gt_poses[_k] - gt_poses[_k - 1]) + CPose2D(_rng.normal(0, 0.02), _rng.normal(0, 0.02), _rng.normal(0, np.radians(1)))
        odometry_and_scans.append((odometry_action(_odometry), scan_frame(ranges[_k])))
    return MAX_RANGE, gt_poses, odometry_and_scans, ranges


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Running the particle filter

    Each call to `CParticleFilter.executeOn(pdf, action, frame)` runs one
    step of MCL: every particle is moved by a random sample of the motion
    model (**prediction**), weighted by how well the scan fits the map from
    that pose (**update**), and the set is **resampled** so that unlikely
    particles die and likely ones multiply. With KLD sampling, the number of
    particles adapts to how spread out the belief is.
    """)
    return


@app.cell
def _(
    CMonteCarloLocalization2D,
    CParticleFilter,
    Randomize,
    TParticleFilterAlgorithm,
    TPose2D,
    adaptive,
    grid,
    init_mode,
    metric_map,
    n_particles,
    np,
    odometry_and_scans,
    path,
    seed,
):
    Randomize(seed.value)  # the filter uses MRPT's random generator
    _rng = np.random.default_rng(seed.value)

    mcl = CMonteCarloLocalization2D()
    mcl.options.metricMap = metric_map
    mcl.options.KLD_params.KLD_maxSampleSize = n_particles.value
    mcl.options.KLD_params.KLD_minSampleSize = min(150, n_particles.value)
    if init_mode.value == "Global localization":
        mcl.resetUniformFreeSpace(grid, 0.7, n_particles.value)
    else:
        _x, _y, _phi = path[0]
        mcl.resetAroundSetOfPoses([TPose2D(_x, _y, _phi)], n_particles.value, 0.3, 0.3, np.radians(10))

    pf = CParticleFilter()
    pf.options.PF_algorithm = TParticleFilterAlgorithm.StandardProposal
    pf.options.adaptiveSampleSize = adaptive.value

    def snapshot():
        """Keeps what is needed to draw the current belief."""
        _p = mcl.getParticlesAsNumpy()  # columns: x, y, phi, log-weight
        # Particles may have different weights if the filter did not resample in this step,
        # so draw (up to) 5000 of them with probability proportional to their weights:
        _w = np.exp(_p[:, 3] - _p[:, 3].max())
        _shown = _rng.choice(len(_p), min(len(_p), 5000), p=_w / _w.sum())
        _cov, _mean = mcl.getCovarianceAndMean()
        return {"particles": _p[_shown, :3], "count": len(mcl), "mean": _mean, "cov": np.array(_cov)}

    history = [snapshot()]
    for _action, _frame in odometry_and_scans:
        pf.executeOn(mcl, _action, _frame)
        history.append(snapshot())
    return (history,)


@app.cell(hide_code=True)
def _(gt_poses, history, mo, np):
    errors = np.array([np.hypot(h["mean"].x - g.x, h["mean"].y - g.y) for h, g in zip(history, gt_poses)])
    yaw_errors = np.degrees([abs((h["mean"] - g).phi) for h, g in zip(history, gt_poses)])
    counts = np.array([h["count"] for h in history])

    step = mo.ui.slider(start=0, stop=len(history) - 1, step=1, value=4, label="Time step", show_value=True, full_width=True)
    step
    return counts, errors, step, yaw_errors


@app.cell(hide_code=True)
def _(counts, errors, mo, step, yaw_errors):
    _k = step.value
    _ok = errors[_k] < 0.3 and yaw_errors[_k] < 5.0
    mo.callout(
        mo.md(
            f"**Step {_k}**: {counts[_k]} particles. Error of the mean estimate: "
            f"**{100 * errors[_k]:.0f} cm**, {yaw_errors[_k]:.1f}° "
            + ("(localized)." if _ok else "(not localized).")
        ),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(Ellipse, MAX_RANGE, grid, gt_poses, history, np, path, plt, ranges, step):
    def draw_pose(ax, pose, color, label):
        ax.annotate("", xy=(pose.x + 0.7 * np.cos(pose.phi), pose.y + 0.7 * np.sin(pose.phi)), xytext=(pose.x, pose.y), arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        ax.plot(pose.x, pose.y, "o", color=color, ms=7, label=label)

    _k = step.value
    _h = history[_k]
    _mean = _h["mean"]

    fig, ax = plt.subplots(figsize=(12, 7.4))
    ax.imshow(grid.getAsNumpy(), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest", extent=[grid.getXMin(), grid.getXMax(), grid.getYMin(), grid.getYMax()])
    ax.plot(path[: _k + 1, 0], path[: _k + 1, 1], "--", color="0.3", lw=1)
    ax.scatter(_h["particles"][:, 0], _h["particles"][:, 1], s=2, color="tab:red", alpha=0.3, zorder=2)
    ax.scatter([], [], s=12, color="tab:red", label="Particles")

    # The laser scan, drawn from the estimated pose: it fits the walls once localized.
    _ok = ranges[_k] < MAX_RANGE
    _ang = np.linspace(-np.pi, np.pi, len(ranges[_k]))[_ok] + _mean.phi
    ax.scatter(_mean.x + ranges[_k][_ok] * np.cos(_ang), _mean.y + ranges[_k][_ok] * np.sin(_ang), s=4, color="tab:green", zorder=3)
    ax.scatter([], [], s=12, color="tab:green", label="Scan, from the estimate")

    # 95% confidence ellipse of the estimated position:
    _vals, _vecs = np.linalg.eigh(_h["cov"][:2, :2])  # ascending eigenvalues
    _minor, _major = 2 * np.sqrt(5.991 * np.maximum(_vals, 0))
    _angle = np.degrees(np.arctan2(_vecs[1, 1], _vecs[0, 1]))  # direction of the major axis
    ax.add_patch(Ellipse((_mean.x, _mean.y), _major, _minor, angle=_angle, fill=False, color="tab:blue", lw=2, zorder=4))
    draw_pose(ax, gt_poses[_k], "black", "True pose")
    draw_pose(ax, _mean, "tab:blue", "Estimate (mean, 95% ellipse)")

    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=4, fontsize=9, frameon=False)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(counts, errors, plt, step):
    fig2, (ax_err, ax_n) = plt.subplots(1, 2, figsize=(12, 3))
    ax_err.semilogy(errors, color="tab:blue")
    ax_err.axhline(0.3, color="tab:green", ls="--", lw=1, label="30 cm")
    ax_err.set_ylabel("Position error (m)")
    ax_err.legend(fontsize=8, loc="upper right")
    ax_n.semilogy(counts, color="tab:red")
    ax_n.set_ylabel("Number of particles")
    for _ax in (ax_err, ax_n):
        _ax.axvline(step.value, color="0.3", lw=1)
        _ax.set_xlabel("Time step")
        _ax.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Things to try**

    - In global localization, the particles first survive in all the places
      that look alike (rooms with a similar shape, the corridor), and the mean
      is meaningless. They collapse into one cluster once the robot sees a
      distinctive part of the office.
    - With few particles (1000 to 3000), often no particle starts close
      enough to the true pose and the filter converges to a wrong place:
      change the random seed to see it. Tracking from a known pose works even
      with 300 particles.
    - With KLD sampling, the number of particles drops by an order of
      magnitude once the belief becomes concentrated, which makes each step
      much faster.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/monte_carlo_localization.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
