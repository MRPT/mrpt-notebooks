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
    # Trajectory families and TP-Space (PTGs)

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to explore **Parameterized Trajectory Generators**
    (PTGs), the core of MRPT's reactive navigator. A PTG is a family of
    kinematically feasible paths that start at the robot, indexed by one
    parameter $\alpha \in (-\pi, \pi]$. Any point $(x, y)$ reached by those
    paths gets new coordinates $(\alpha, d)$: *which* path reaches it, and
    *after how many meters*. In this **TP-Space** (Trajectory Parameter
    Space) every path becomes a straight radial line, so a differential-drive
    robot can be treated as a point that moves freely in any direction, and
    obstacle avoidance becomes much simpler.

    It uses the `mrpt.nav` module: `CParameterizedTrajectoryGenerator` (its
    factory `CreatePTG()`, `initialize()` and the Workspace to TP-Space map
    `inverseMap_WS2TP()`), with the PTG classes `CPTG_DiffDrive_C`,
    `CPTG_DiffDrive_alpha`, `CPTG_DiffDrive_CS` and `CPTG_Holo_Blend`.

    **Move the sliders**: change the PTG family, the robot speed limits, the
    number of paths, or drag the wall around and watch how it maps into TP-Space.
    """)
    return


@app.cell
def _():
    import contextlib
    import os
    import sys
    import tempfile

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    from mrpt.nav import CParameterizedTrajectoryGenerator
    return (
        CParameterizedTrajectoryGenerator,
        LinearSegmentedColormap,
        contextlib,
        mo,
        np,
        os,
        plt,
        sys,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    ptg_class = mo.ui.dropdown(
        options={
            "C: circular arcs": "CPTG_DiffDrive_C",
            "α-A: heading converges to α": "CPTG_DiffDrive_alpha",
            "CS: arc, then straight line": "CPTG_DiffDrive_CS",
            "Holonomic robot (blended velocities)": "CPTG_Holo_Blend",
        },
        value="α-A: heading converges to α",
        label="PTG family",
    )
    backwards = mo.ui.checkbox(label="Drive backwards (K = -1)")
    v_max = mo.ui.slider(start=0.5, stop=2.0, step=0.1, value=1.0, label="Max. linear speed (m/s)", show_value=True)
    w_max = mo.ui.slider(start=30, stop=180, step=10, value=60, label="Max. angular speed (deg/s)", show_value=True)
    num_paths = mo.ui.slider(start=11, stop=61, step=2, value=31, label="Number of paths (α resolution)", show_value=True)
    ref_dist = mo.ui.slider(start=2.0, stop=4.0, step=0.5, value=3.0, label="Reference distance (m)", show_value=True)

    mo.vstack(
        [
            mo.md("### PTG parameters"),
            mo.hstack([ptg_class, backwards], justify="start", gap=2),
            mo.hstack([v_max, w_max], justify="start", gap=2),
            mo.hstack([num_paths, ref_dist], justify="start", gap=2),
        ]
    )
    return backwards, num_paths, ptg_class, ref_dist, v_max, w_max


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Create the PTG

    PTGs are created by class name with a factory, from parameters in MRPT
    INI config-file format. The key ones are the robot speed limits, the
    number of discrete paths (the resolution in $\alpha$) and the
    *reference distance*: paths are only simulated up to that length, which
    is then used to normalize TP-Space distances to $[0, 1]$.
    `initialize()` simulates the paths and builds look-up tables (cached on
    disk, so it only takes a while the first time for each set of parameters).
    """)
    return


@app.cell
def _(contextlib, os, sys, tempfile):
    cache_dir = tempfile.mkdtemp(prefix="mrpt_ptg_cache_")

    def initialize_quietly(ptg):
        """Calls ptg.initialize(), which saves its look-up tables to the current
        directory and prints progress from C++: use a temporary directory, and hide the output."""
        sys.stdout.flush()
        saved_stdout = os.dup(1)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        try:
            with contextlib.chdir(cache_dir):
                ptg.initialize()
        finally:
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)
            os.close(devnull)
    return (initialize_quietly,)


@app.cell
def _(
    CParameterizedTrajectoryGenerator,
    backwards,
    initialize_quietly,
    num_paths,
    ptg_class,
    ref_dist,
    v_max,
    w_max,
):
    # Robot footprint (m), in the robot frame (x: forward, y: left):
    robot_shape = [(-0.2, 0.3), (0.5, 0.3), (0.5, -0.3), (-0.2, -0.3)]

    ini_text = "\n".join(
        [
            "[PTG]",
            f"refDistance = {ref_dist.value}",
            f"num_paths = {num_paths.value}",
            f"v_max_mps = {v_max.value}",
            f"w_max_dps = {w_max.value}",
            f"K = {-1 if backwards.value else 1}",
            "resolution = 0.25     # cell size of the collision look-up table (m)",
            "cte_a0v_deg = 57      # only for alpha-A",
            "cte_a0w_deg = 57      # only for alpha-A",
            "T_ramp_max = 0.8      # only for holonomic",
            "robot_radius = 0.35   # only for holonomic",
        ]
        + [f"shape_x{i} = {x}\nshape_y{i} = {y}" for i, (x, y) in enumerate(robot_shape)]
    )

    ptg = CParameterizedTrajectoryGenerator.CreatePTG(ptg_class.value, ini_text, "PTG")
    initialize_quietly(ptg)
    return ini_text, ptg, robot_shape


@app.cell(hide_code=True)
def _(ini_text, mo, ptg):
    mo.hstack(
        [
            mo.md(f"```ini\n{ini_text}\n```"),
            mo.md(
                f"""
    `ptg.getDescription()`: **{ptg.getDescription()}**

    `ptg.isInitialized()`: **{ptg.isInitialized()}**

    `ptg.getPathCount()`: **{ptg.getPathCount()}** paths
    """
            ),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. From Workspace to TP-Space

    `inverseMap_WS2TP(x, y)` answers the key question of TP-Space
    navigation: which path $k$ (that is, which $\alpha$) passes through the
    point $(x, y)$, and at which normalized distance $d$ along it? We call it
    for every point of a grid around the robot. Then, the **curves of
    constant $k$ are the trajectories** of the family (left plot, drawn where
    $k$ changes from one path to the next), and the curves of constant $d$ are
    the points the robot reaches after traveling the same distance. In
    TP-Space (right plot) the same paths are just straight rays. Colors
    encode $\alpha$: blue paths turn right, gray ones go straight ahead, and
    red ones turn left.

    The wall below is an obstacle. Each of its points is also mapped to
    TP-Space, and every path is cut at the first obstacle it meets: this
    free distance per direction is what MRPT feeds to a *holonomic*
    obstacle avoidance method (e.g. ND or VFF) to pick the best $\alpha$.
    For simplicity the robot is treated as a point here (MRPT's navigator also
    accounts for its footprint).
    """)
    return


@app.cell
def _(np, ptg, ref_dist):
    # TP-Space coordinates (path index k, normalized distance d) of each Workspace grid point.
    # An even number of cells avoids the robot origin (0, 0), which is not part of TP-Space.
    grid = np.linspace(-1.05, 1.05, 240) * ref_dist.value
    grid_k = np.full((grid.size, grid.size), np.nan)
    grid_d = np.full((grid.size, grid.size), np.nan)
    _inside = np.hypot(*np.meshgrid(grid, grid)) <= ref_dist.value
    for _iy, _ix in zip(*np.nonzero(_inside)):
        _tp = ptg.inverseMap_WS2TP(grid[_ix], grid[_iy], tolerance_dist=0.25)
        if _tp is not None and _tp[1] <= 1.0:
            grid_k[_iy, _ix], grid_d[_iy, _ix] = _tp
    return grid, grid_d, grid_k


@app.cell(hide_code=True)
def _(mo):
    wall_x = mo.ui.slider(start=-3.0, stop=3.0, step=0.1, value=1.6, label="Wall center: x (m)", show_value=True)
    wall_y = mo.ui.slider(start=-3.0, stop=3.0, step=0.1, value=0.5, label="y (m)", show_value=True)
    wall_len = mo.ui.slider(start=0.2, stop=3.0, step=0.1, value=1.5, label="Wall length (m)", show_value=True)
    wall_angle = mo.ui.slider(start=0, stop=180, step=5, value=70, label="Wall angle (deg)", show_value=True)

    mo.vstack(
        [
            mo.md("### Obstacle"),
            mo.hstack([wall_x, wall_y], justify="start", gap=2),
            mo.hstack([wall_len, wall_angle], justify="start", gap=2),
        ]
    )
    return wall_angle, wall_len, wall_x, wall_y


@app.cell
def _(np, ptg, wall_angle, wall_len, wall_x, wall_y):
    # The wall, sampled as obstacle points every ~1 cm:
    _s = np.linspace(-0.5, 0.5, int(wall_len.value * 100) + 1) * wall_len.value
    _a = np.radians(wall_angle.value)
    wall = np.column_stack([wall_x.value + _s * np.cos(_a), wall_y.value + _s * np.sin(_a)])

    # Map each obstacle point to TP-Space, and keep the closest obstacle along each path:
    n_paths = ptg.getPathCount()
    free_d = np.ones(n_paths)  # collision-free distance of each path (1 = the whole path)
    obstacles_tp = []
    for _x, _y in wall:
        if np.hypot(_x, _y) < 1e-6:
            continue  # the origin itself is not part of TP-Space
        _tp = ptg.inverseMap_WS2TP(_x, _y, tolerance_dist=0.05)
        if _tp is not None and _tp[1] <= 1.0:
            _k, _d = _tp
            obstacles_tp.append((_k, _d))
            free_d[_k] = min(free_d[_k], _d)
    obstacles_tp = np.array(obstacles_tp).reshape(-1, 2)

    # Direction alpha of each path k (same formula as the C++ index2alpha()):
    alphas = np.pi * (-1.0 + 2.0 * (np.arange(n_paths) + 0.5) / n_paths)
    return alphas, free_d, n_paths, obstacles_tp, wall


@app.cell(hide_code=True)
def _(alphas, free_d, mo, n_paths, np, ref_dist):
    _blocked = free_d < 1.0
    _k = int(np.argmin(free_d))
    mo.md(
        f"The wall blocks **{_blocked.sum()} of {n_paths}** paths. The closest obstacle along a path is "
        f"at **{free_d.min() * ref_dist.value:.2f} m** (path k = {_k}, α = {np.degrees(alphas[_k]):.0f}°)."
        if _blocked.any()
        else f"The wall does not block any of the {n_paths} paths (within the reference distance)."
    )
    return


@app.cell(hide_code=True)
def _(
    LinearSegmentedColormap,
    alphas,
    free_d,
    grid,
    grid_d,
    grid_k,
    n_paths,
    np,
    obstacles_tp,
    plt,
    ref_dist,
    robot_shape,
    wall,
):
    # Color = alpha: blue for right turns, gray for straight ahead, red for left turns.
    cmap = LinearSegmentedColormap.from_list("alpha", ["tab:blue", "0.45", "tab:red"])
    levels = np.arange(n_paths - 1) + 0.5  # boundaries between consecutive paths
    level_colors = cmap(levels / (n_paths - 1))

    # A 3x3 box filter on k, for smoother contour lines. Skip cells next to the jump
    # from the last path to the first one (alpha = +-pi), which is not a real boundary:
    _kp = np.pad(grid_k, 1, mode="edge")
    _nbrs = np.stack([_kp[i : i + grid.size, j : j + grid.size] for i in range(3) for j in range(3)])
    _k_smooth = np.where(np.ptp(_nbrs, axis=0) < n_paths / 2, _nbrs.mean(axis=0), np.nan)

    # Split the grid into the collision-free and blocked parts of each path:
    _valid = ~np.isnan(grid_k)
    _k = np.where(_valid, grid_k, 0).astype(int)
    _free = _valid & (grid_d <= free_d[_k])
    _blocked = _valid & ~_free

    fig = plt.figure(figsize=(12, 5.8))
    ax_ws = fig.add_subplot(1, 2, 1)
    ax_tp = fig.add_subplot(1, 2, 2, projection="polar")

    # Workspace:
    ax_ws.contour(grid, grid, np.where(_blocked, _k_smooth, np.nan), levels=levels, colors="0.85", linewidths=0.8)
    ax_ws.contour(grid, grid, np.where(_free, _k_smooth, np.nan), levels=levels, colors=level_colors, linewidths=1.0)
    _cs = ax_ws.contour(
        grid, grid, grid_d * ref_dist.value, levels=np.arange(1.0, ref_dist.value, 1.0),
        colors="0.4", linestyles=":", linewidths=1.0,
    )
    ax_ws.clabel(_cs, fmt="%.0f m", fontsize=8)
    ax_ws.fill(*np.array(robot_shape).T, color="0.5", alpha=0.8, zorder=3, label="Robot")
    ax_ws.plot(*wall.T, color="black", lw=4, solid_capstyle="round", label="Wall (obstacle)")
    ax_ws.plot([], [], color="0.85", label="Blocked part of the paths")
    ax_ws.plot([], [], color="0.4", ls=":", label="Same distance along the paths")
    _lim = 1.05 * ref_dist.value
    ax_ws.set(xlim=(-_lim, _lim), ylim=(-_lim, _lim), xlabel="x (m)", ylabel="y (m)", title="Workspace")
    ax_ws.set_aspect("equal")
    ax_ws.grid(alpha=0.3)
    ax_ws.legend(loc="lower left", fontsize=8)

    # TP-Space: one ray per path, colored up to its first obstacle:
    for _a, _d, _c in zip(alphas, free_d, cmap(np.linspace(0, 1, n_paths))):
        ax_tp.plot([_a, _a], [0, _d], color=_c, lw=1.2)
        ax_tp.plot([_a, _a], [_d, 1], color="0.85", lw=1.2)
    if len(obstacles_tp):
        ax_tp.scatter(alphas[obstacles_tp[:, 0].astype(int)], obstacles_tp[:, 1], s=8, color="black", zorder=3, label="Wall points")
        ax_tp.legend(loc="upper left", bbox_to_anchor=(-0.12, 1.08), fontsize=8)
    ax_tp.set_ylim(0, 1)
    ax_tp.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_tp.tick_params(axis="y", labelsize=8)
    ax_tp.set_title("TP-Space: angle α, radius d (normalized)")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to try:

    - **C** paths are circular arcs with radius proportional to $1/\alpha$; the
      tightest turn radius is $v_{max} / \omega_{max}$, so points inside the
      two tightest circles cannot be reached.
    - **α-A** paths turn quickly towards heading $\alpha$ and then go straight,
      so they cover the space around and behind the robot.
    - Reduce the number of paths to see the coarser TP-Space discretization:
      a narrow wall may then fall *between* two paths.
    - Enable **Drive backwards** to get the mirrored family used for reversing
      (differential-drive PTGs only).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/ptg_trajectories.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
