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
    # Occupancy grid mapping from laser scans

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to build an occupancy grid map of a simulated office
    from the 2D laser scans of a robot whose path is known. It uses three MRPT
    modules:

    - `mrpt.obs`: the laser scan observation (`CObservation2DRangeScan`).
    - `mrpt.poses`: the robot pose at each scan (`CPose3D`).
    - `mrpt.maps`: the occupancy grid (`COccupancyGridMap2D`), updated with `insertObservation()`.

    An occupancy grid divides the floor into square cells, each one storing
    the probability of being occupied by an obstacle. Every laser ray is
    evidence that the cells it crosses are free, and that the cell where it
    ends is occupied. **Move the sliders** to insert more scans, add range
    noise, or change the cell size.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mrpt.maps import COccupancyGridMap2D
    from mrpt.obs import CObservation2DRangeScan
    from mrpt.poses import CPose3D
    return COccupancyGridMap2D, CObservation2DRangeScan, CPose3D, mo, np, plt


@app.cell(hide_code=True)
def _(mo, path):
    n_scans = mo.ui.slider(start=1, stop=len(path), step=1, value=len(path), label="Scans inserted", show_value=True)
    noise = mo.ui.slider(start=0.0, stop=20.0, step=0.5, value=2.0, label="Range noise σ (cm)", show_value=True)
    resolution = mo.ui.slider(steps=[0.05, 0.1, 0.15, 0.2, 0.25], value=0.1, label="Cell size (m)", show_value=True)
    show_walls = mo.ui.checkbox(value=False, label="Show the true walls")

    mo.vstack(
        [
            mo.md("### Parameters"),
            mo.hstack([n_scans, noise], justify="start", gap=2),
            mo.hstack([resolution, show_walls], justify="start", gap=2),
        ]
    )
    return n_scans, noise, resolution, show_walls


@app.cell
def _(np):
    # The simulated office: wall segments [x0, y0, x1, y1], in meters.
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

    # Robot path: a pose every 40 cm along these waypoints, heading forward.
    _waypoints = [(1.8, 3.2), (3, 4), (3, 6), (8.5, 6), (8.5, 9), (8.5, 6), (10.5, 6), (10.5, 2.5), (10.5, 6), (16, 6), (16, 9.5)]
    _xy = [np.array(_waypoints[0], float)]
    for _a, _b in zip(_waypoints[:-1], _waypoints[1:]):
        _n = max(1, round(np.hypot(_b[0] - _a[0], _b[1] - _a[1]) / 0.4))
        _xy += [np.array(_a) + (np.array(_b) - np.array(_a)) * k / _n for k in range(1, _n + 1)]
    _xy = np.array(_xy)
    _heading = np.arctan2(np.diff(_xy[:, 1]), np.diff(_xy[:, 0]))
    path = np.column_stack([_xy, np.append(_heading, _heading[-1])])  # (x, y, phi) per scan
    return path, simulate_ranges, walls


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Simulated laser scans

    A `CObservation2DRangeScan` stores one range per ray, a validity flag per
    ray (no echo within `maxRange` means invalid), and the scan geometry:
    `aperture` (here 360°), `rightToLeft` and the sensor pose on the robot.
    The ranges come from ray-casting against the office walls, plus Gaussian
    noise.
    """)
    return


@app.cell
def _(CObservation2DRangeScan, noise, np, path, simulate_ranges):
    MAX_RANGE = 15.0  # meters

    def make_scan(scan_ranges):
        """Wraps an array of ranges into an MRPT scan observation."""
        obs = CObservation2DRangeScan()
        obs.aperture = 2 * np.pi
        obs.maxRange = MAX_RANGE
        obs.resizeScan(len(scan_ranges))
        for i, r in enumerate(scan_ranges):
            valid = bool(r < MAX_RANGE)
            obs.setScanRange(i, float(r) if valid else MAX_RANGE)
            obs.setScanRangeValidity(i, valid)
        return obs

    _rng = np.random.default_rng(0)
    ranges = [simulate_ranges(pose, _rng, noise.value / 100) for pose in path]
    scans = [make_scan(r) for r in ranges]
    return MAX_RANGE, ranges, scans


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Inserting the scans into the grid

    `insertObservation(scan, robot_pose)` updates every cell touched by the
    rays of the scan, taken from the given robot pose. Each cell keeps its
    occupancy probability as **log-odds**, so a Bayesian update is just an
    addition: consistent observations accumulate, contradicting ones cancel
    out. `getAsNumpy()` returns the map as an image (1: free, 0: occupied,
    0.5: never observed).
    """)
    return


@app.cell
def _(COccupancyGridMap2D, CPose3D, n_scans, path, resolution, scans):
    # Three cells to follow during mapping: a wall, the corridor, and a room visited at the end.
    probes = {"Wall": (6.5, 2.5), "Corridor floor": (6.0, 6.0), "Last room floor": (18.5, 10.5)}

    def occupancy_near(grid, x, y):
        """Occupancy probability of the most occupied cell around (x, y), as walls may lie on cell borders."""
        _r = grid.getResolution()
        return 1 - min(grid.getPos(x + i * _r, y + j * _r) for i in (-1, 0, 1) for j in (-1, 0, 1))

    # The grid has a margin around the building so it does not need to grow.
    grid = COccupancyGridMap2D(-4.0, 24.0, -4.0, 16.0, resolution.value)
    history = {name: [] for name in probes}
    for _scan, (_x, _y, _phi) in zip(scans[: n_scans.value], path):
        grid.insertObservation(_scan, CPose3D(_x, _y, 0.0, _phi, 0.0, 0.0))
        for _name, (_px, _py) in probes.items():
            history[_name].append(occupancy_near(grid, _px, _py))

    occupancy_map = grid.getAsNumpy()
    return grid, history, occupancy_map, probes


@app.cell(hide_code=True)
def _(grid, mo, np, occupancy_map):
    # Fraction of the building floor (0 < x < 20, 0 < y < 12) observed so far:
    _xs = grid.getXMin() + grid.getResolution() * (np.arange(grid.getSizeX()) + 0.5)
    _ys = grid.getYMin() + grid.getResolution() * (np.arange(grid.getSizeY()) + 0.5)
    _inside = ((_xs > 0) & (_xs < 20))[None, :] & ((_ys > 0) & (_ys < 12))[:, None]
    _observed = np.abs(occupancy_map[_inside] - 0.5) > 0.01
    mo.md(
        f"Grid of **{grid.getSizeX()} × {grid.getSizeY()}** cells of {100 * grid.getResolution():.0f} cm. "
        f"Observed so far: **{100 * _observed.mean():.0f}%** of the building floor."
    )
    return


@app.cell(hide_code=True)
def _(
    MAX_RANGE,
    grid,
    n_scans,
    np,
    occupancy_map,
    path,
    plt,
    probes,
    ranges,
    show_walls,
    walls,
):
    probe_colors = {"Wall": "tab:orange", "Corridor floor": "tab:green", "Last room floor": "tab:purple"}

    fig, ax = plt.subplots(figsize=(12, 7.6))
    ax.imshow(
        occupancy_map,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        interpolation="nearest",
        extent=[grid.getXMin(), grid.getXMax(), grid.getYMin(), grid.getYMax()],
    )
    if show_walls.value:
        for _x0, _y0, _x1, _y1 in walls:
            ax.plot([_x0, _x1], [_y0, _y1], color="tab:cyan", lw=1)

    # Robot path so far, and the last scan:
    _k = n_scans.value - 1
    _x, _y, _phi = path[_k]
    ax.plot(path[: _k + 1, 0], path[: _k + 1, 1], ".-", color="tab:blue", ms=4, lw=1, label="Robot path")
    _ok = ranges[_k] < MAX_RANGE
    _ang = np.linspace(-np.pi, np.pi, len(ranges[_k]))[_ok] + _phi
    ax.scatter(_x + ranges[_k][_ok] * np.cos(_ang), _y + ranges[_k][_ok] * np.sin(_ang), s=4, color="tab:red", label="Last scan", zorder=3)
    ax.annotate("", xy=(_x + 0.8 * np.cos(_phi), _y + 0.8 * np.sin(_phi)), xytext=(_x, _y), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.5))

    for _name, (_px, _py) in probes.items():
        ax.plot(_px, _py, "o", ms=11, mfc="none", mew=2.5, color=probe_colors[_name])
        ax.annotate(_name, (_px, _py), xytext=(9, 9), textcoords="offset points", color=probe_colors[_name], fontsize=9, fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    fig
    return (probe_colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. How a cell becomes certain

    Each scan can change a cell only up to a maximum certainty (65% by
    default), so a cell needs several consistent observations to become
    confidently free or occupied. Below, the occupancy probability of the
    three cells circled in the map, as scans are inserted. Cells never reached
    by any ray, like the inside of furniture, stay at 0.5 (unknown).
    """)
    return


@app.cell(hide_code=True)
def _(history, path, plt, probe_colors):
    fig2, ax2 = plt.subplots(figsize=(12, 3.2))
    for _name, _values in history.items():
        ax2.plot(range(1, len(_values) + 1), _values, ".-", color=probe_colors[_name], label=_name)
    ax2.axhline(0.5, color="0.6", ls="--", lw=1)
    ax2.text(1, 0.53, "unknown", color="0.5", fontsize=8)
    ax2.set_xlim(0, len(path) + 1)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_xlabel("Scans inserted")
    ax2.set_ylabel("Occupancy probability")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="center right", fontsize=8)
    fig2.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/occupancy_grid_mapping.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
