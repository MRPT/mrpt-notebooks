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
    # Pose graphs and loop closures

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to build the pose graph of a robot that drives
    through a small city-block route and revisits some places. It uses two
    MRPT modules:

    - `mrpt.poses`: SE(2) poses (`CPose2D`), their composition `a + b` and relative pose `a - b`.
    - `mrpt.graphs`: the pose graph (`CNetworkOfPoses2D`), its spanning-tree estimate and shortest paths.

    In graph SLAM, each robot pose is a **node** and each relative measurement
    is an **edge**: odometry links consecutive poses, and a **loop closure**
    links two poses at the same place, visited at different times. Odometry
    alone drifts without bound; loop closures are what tie the trajectory
    back together. **Move the sliders**: everything below is recomputed.
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mrpt.graphs import CNetworkOfPoses2D
    from mrpt.poses import CPose2D
    return CNetworkOfPoses2D, CPose2D, Path, mo, np, plt, tempfile


@app.cell(hide_code=True)
def _(mo):
    noise_xy = mo.ui.slider(start=0.0, stop=10.0, step=0.5, value=3.0, label="Odometry noise: xy σ (cm/m)", show_value=True)
    noise_phi = mo.ui.slider(start=0.0, stop=3.0, step=0.1, value=1.0, label="yaw σ (deg/m)", show_value=True)
    n_loops = mo.ui.slider(start=0, stop=24, step=1, value=6, label="Number of loop closures", show_value=True)
    seed = mo.ui.slider(start=0, stop=20, step=1, value=1, label="Random seed", show_value=True)

    mo.vstack(
        [
            mo.md("### Parameters"),
            mo.hstack([noise_xy, noise_phi], justify="start", gap=2),
            mo.hstack([n_loops, seed], justify="start", gap=2),
        ]
    )
    return n_loops, noise_phi, noise_xy, seed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Ground truth and simulated measurements

    The robot drives along the streets of a 3 × 2 block grid (10 m blocks),
    recording a pose every meter. Odometry measures the relative pose
    between consecutive poses, `gt[i + 1] - gt[i]`, corrupted by noise.
    A loop closure is the relative pose between two poses close in space but
    far apart in time, here with a small error (5 cm, 1°), as if obtained by
    matching two laser scans.
    """)
    return


@app.cell
def _(CPose2D, np):
    # Street corners visited by the robot, in meters:
    corners = [(0, 0), (30, 0), (30, 20), (0, 20), (0, 0), (10, 0), (10, 20), (20, 20), (20, 0), (30, 0), (30, 10), (0, 10)]

    # Ground truth: one pose every meter, heading along the street.
    xy = [np.array(corners[0], float)]
    for _a, _b in zip(corners[:-1], corners[1:]):
        _n = int(np.hypot(_b[0] - _a[0], _b[1] - _a[1]))
        xy += [np.array(_a) + (np.array(_b) - np.array(_a)) * k / _n for k in range(1, _n + 1)]
    xy = np.array(xy)
    heading = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
    heading = np.append(heading, heading[-1])
    gt = [CPose2D(x, y, h) for (x, y), h in zip(xy, heading)]
    n_nodes = len(gt)

    # Candidate loop closures: pairs of poses at the same place, at least 15 steps apart.
    _dist = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    _i, _j = np.nonzero((_dist < 0.3) & (np.arange(n_nodes)[None, :] - np.arange(n_nodes)[:, None] > 15))
    candidates = sorted(zip(_i.tolist(), _j.tolist()), key=lambda p: p[1])
    return candidates, gt, n_nodes, xy


@app.cell
def _(CPose2D, candidates, gt, n_loops, n_nodes, noise_phi, noise_xy, np, seed):
    rng = np.random.default_rng(seed.value)

    def noisy(pose, sigma_xy, sigma_phi):
        """Composes a pose with a random Gaussian error, in its local frame."""
        return pose + CPose2D(rng.normal(0, sigma_xy), rng.normal(0, sigma_xy), rng.normal(0, sigma_phi))

    # Odometry edges (i, i+1): 1 m steps, so the noise sliders are per meter.
    odometry = [
        (i, i + 1, noisy(gt[i + 1] - gt[i], noise_xy.value / 100, np.radians(noise_phi.value)))
        for i in range(n_nodes - 1)
    ]
    # Loop closure edges, evenly spread over the candidates:
    _picks = np.linspace(0, len(candidates) - 1, n_loops.value).round().astype(int) if n_loops.value else []
    loops = [(i, j, noisy(gt[j] - gt[i], 0.05, np.radians(1.0))) for i, j in (candidates[k] for k in _picks)]
    return loops, odometry


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The pose graph and its initial estimate

    `CNetworkOfPoses2D.insertEdge(i, j, z)` adds the measurement `z`: the pose
    of node `j` as seen from node `i`. With odometry only, the global pose of
    each node is the composition of all the edges before it (dead reckoning),
    and its error grows with the distance traveled.

    `dijkstra_nodes_estimate()` computes the global pose of every node by
    composing edges along the **shortest path** (fewest edges) from the root
    node 0. Loop closures act as shortcuts: many nodes become fewer edges away
    from the root, so their estimate accumulates less drift. This spanning-tree
    estimate is the usual starting point of graph SLAM optimizers.
    """)
    return


@app.cell
def _(CNetworkOfPoses2D, CPose2D, loops, np, odometry):
    graph = CNetworkOfPoses2D()
    for _i, _j, _z in odometry + loops:
        graph.insertEdge(_i, _j, _z)

    # Dead reckoning: compose the odometry increments one after another.
    dead_reckoning = [CPose2D()]
    for _i, _j, _z in odometry:
        dead_reckoning.append(dead_reckoning[-1] + _z)

    # Spanning-tree estimate, rooted at node 0:
    graph.root = 0
    graph.dijkstra_nodes_estimate()
    tree_estimate = [graph.getNodePose(i) for i in graph.getNodeIDs()]
    hops = graph.getNodeDistances(0)  # node ID -> number of edges to the root

    def residual(i, j, z, poses):
        """Mismatch (m) between edge z and the relative pose of j w.r.t. i in an estimate."""
        _e = (poses[j] - poses[i]) - z
        return np.hypot(_e.x, _e.y)

    # Edges outside the spanning tree are not satisfied by its estimate:
    seams = [(i, j) for i, j, z in odometry + loops if residual(i, j, z, tree_estimate) > 0.01]
    return dead_reckoning, graph, hops, seams, tree_estimate


@app.cell(hide_code=True)
def _(dead_reckoning, graph, gt, hops, loops, mo, np, seams, tree_estimate):
    def rmse(poses):
        return np.sqrt(np.mean([(p.x - g.x) ** 2 + (p.y - g.y) ** 2 for p, g in zip(poses, gt)]))

    mo.md(
        f"""
    The graph has **{graph.nodeCount()}** nodes and **{graph.edgeCount()}** edges
    ({len(loops)} loop closures). The farthest node is **{max(hops.values()):.0f}** edges away
    from the root (it would be {graph.nodeCount() - 1} with odometry alone).

    | Estimate | Position RMSE vs. ground truth (m) |
    |---|---|
    | Dead reckoning (odometry only) | {rmse(dead_reckoning):.2f} |
    | Spanning tree (`dijkstra_nodes_estimate`) | {rmse(tree_estimate):.2f} |

    Edges that the spanning-tree estimate does not satisfy (red in the plot): **{len(seams)}**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, n_nodes):
    target = mo.ui.slider(start=1, stop=n_nodes - 1, step=1, value=n_nodes - 1, label="Shortest path from node 0 to node", show_value=True)
    target
    return (target,)


@app.cell(hide_code=True)
def _(dead_reckoning, graph, hops, loops, np, plt, seams, target, tree_estimate, xy):
    def xy_of(poses):
        return np.array([[p.x, p.y] for p in poses])

    _dr = xy_of(dead_reckoning)
    _te = xy_of(tree_estimate)
    _path = graph.dijkstra_path(0, target.value)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    for ax in axs:
        ax.plot(*xy.T, color="0.8", lw=4, zorder=0, label="Ground truth")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    # Left: dead reckoning, with the loop closures it violates.
    axs[0].plot(*_dr.T, ".-", color="tab:blue", ms=3, lw=1, label="Dead reckoning")
    for _k, (_i, _j, _z) in enumerate(loops):
        axs[0].plot(*_dr[[_i, _j]].T, "--", color="tab:red", lw=1, label="Loop closures" if _k == 0 else None)
    axs[0].set_title("Odometry only: nodes that should coincide drift apart")

    # Right: spanning-tree estimate, colored by distance (in edges) to the root.
    _sc = axs[1].scatter(*_te.T, c=[hops[i] for i in range(len(_te))], cmap="viridis", s=10, zorder=3)
    for _k, (_i, _j) in enumerate(seams):
        axs[1].plot(*_te[[_i, _j]].T, "-", color="tab:red", lw=1.5, label="Unsatisfied edges" if _k == 0 else None)
    axs[1].plot(*_te[_path].T, "-", color="tab:orange", lw=2.5, alpha=0.8, label=f"Path to node {target.value}")
    axs[1].plot(*_te[target.value], "*", color="tab:orange", ms=14, mec="k", zorder=4)
    axs[1].set_title("Spanning-tree estimate (dijkstra_nodes_estimate)")
    fig.colorbar(_sc, ax=axs[1], fraction=0.035, pad=0.02, label="Edges to node 0")

    for ax in axs:
        ax.plot(0, 0, "ks", ms=7, zorder=5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(graph, gt, mo, target):
    _path = graph.dijkstra_path(0, target.value)
    mo.md(
        f"""
    `dijkstra_path(0, {target.value})` returns the shortest path from the root (black square) to the
    selected node (star): **{len(_path) - 1}** edges, instead of {target.value} along the odometry chain.
    Composing these edges gives the pose that `dijkstra_nodes_estimate()` assigned to that node,
    `{graph.getNodePose(target.value)}`, while its ground truth is `{gt[target.value]}`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Where does the error go?

    The spanning tree satisfies exactly the edges it is made of. Every other
    edge (red above) carries all the inconsistency accumulated along the two
    branches of the tree that meet there, and the estimate jumps at those
    places. The position error grows with the number of edges to the root,
    not with the time at which a pose was recorded:
    """)
    return


@app.cell(hide_code=True)
def _(dead_reckoning, gt, hops, np, plt, tree_estimate):
    def errors(poses):
        return [np.hypot(p.x - g.x, p.y - g.y) for p, g in zip(poses, gt)]

    fig2, (ax_e, ax_h) = plt.subplots(2, 1, figsize=(12, 4.2), sharex=True)
    ax_e.plot(errors(dead_reckoning), color="tab:blue", label="Dead reckoning")
    ax_e.plot(errors(tree_estimate), color="tab:green", label="Spanning tree")
    ax_e.set_ylabel("Position error (m)")
    ax_e.legend(fontsize=8, loc="upper left")
    ax_h.plot([0, len(gt) - 1], [0, len(gt) - 1], color="tab:blue", label="Odometry only")
    ax_h.plot([hops[i] for i in range(len(gt))], color="tab:green", label="With loop closures")
    ax_h.set_ylabel("Edges to node 0")
    ax_h.legend(fontsize=8, loc="upper left")
    ax_h.set_xlabel("Node index (time)")
    for _ax in (ax_e, ax_h):
        _ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Next step: optimization

    The final step of graph SLAM is a nonlinear least-squares optimization
    (e.g. Levenberg-Marquardt) that moves all nodes at once to satisfy every
    edge as well as possible, spreading the error over the whole loop instead
    of concentrating it on a few edges. MRPT implements it in C++
    (`mrpt::graphslam::optimize_spa_levmarq`), but the `mrpt3` Python wheel
    does not expose it yet. The graph can be saved in the standard
    [g2o](https://github.com/RainerKuemmerle/g2o) text format, readable by
    MRPT's `graphslam-engine` application and other graph optimizers:
    """)
    return


@app.cell
def _(Path, graph, mo, tempfile):
    with tempfile.TemporaryDirectory() as _tmp:
        _file = Path(_tmp) / "pose_graph.g2o"
        graph.saveToTextFile(str(_file))
        g2o_lines = _file.read_text().splitlines()

    _vertices = [line for line in g2o_lines if line.startswith("VERTEX")]
    _edges = [line for line in g2o_lines if line.startswith("EDGE")]
    _excerpt = "\n".join(_vertices[:3] + ["..."] + _edges[:3] + ["..."])
    mo.md(f"`pose_graph.g2o`: {len(_vertices)} vertices, {len(_edges)} edges.\n\n```\n{_excerpt}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/pose_graph_slam.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
