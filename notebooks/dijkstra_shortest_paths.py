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
    # Shortest paths in graphs (Dijkstra)

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to find shortest paths in a **topological map**: a
    graph whose nodes are places and whose edges join places that can be
    traveled between directly. It uses two MRPT modules:

    - `mrpt.poses`: SE(2) poses (`CPose2D`) and the relative pose between two of them, `b - a`.
    - `mrpt.graphs`: the graph (`CNetworkOfPoses2D`) and Dijkstra's algorithm on it
      (`dijkstra_path`, `dijkstra_tree`, `getNodeDistances`).

    Dijkstra's algorithm finds the cheapest path from a source node to every
    other node, given a non-negative **cost** for each edge. The cost decides
    what "shortest" means: with one per edge, the path with fewest edges;
    with the length of each edge, the path with the shortest distance.
    **Move the sliders**: everything below is recomputed.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mrpt.graphs import CNetworkOfPoses2D
    from mrpt.poses import CPose2D
    return CNetworkOfPoses2D, CPose2D, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    n_nodes = mo.ui.slider(start=5, stop=60, step=1, value=20, label="Number of nodes", show_value=True)
    radius = mo.ui.slider(start=4.0, stop=15.0, step=0.5, value=12.0, label="Link radius (m)", show_value=True)
    seed = mo.ui.slider(start=0, stop=20, step=1, value=4, label="Random seed", show_value=True)
    cost = mo.ui.radio(
        options=["Fewest edges", "Metric distance"], value="Metric distance", label="Edge cost", inline=True
    )

    mo.vstack(
        [
            mo.md("### Parameters"),
            mo.hstack([n_nodes, radius, seed], justify="start", gap=2),
            cost,
        ]
    )
    return cost, n_nodes, radius, seed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. A random topological map

    The nodes are random poses in a 30 m × 30 m area, and every pair of nodes
    closer than the link radius is joined by an edge. As in a pose graph, an
    edge `i → j` stores the pose of node `j` relative to node `i`; its length
    is `edge.norm()`. Dijkstra in `mrpt.graphs` ignores the direction of the
    edges, so one edge per pair is enough.
    """)
    return


@app.cell
def _(CNetworkOfPoses2D, CPose2D, mo, n_nodes, np, radius, seed):
    # Custom edge costs and dijkstra_tree() are newer than mrpt3 3.6.3:
    mo.stop(
        not hasattr(CNetworkOfPoses2D, "dijkstra_tree"),
        mo.callout(mo.md("This notebook needs a newer `mrpt3`: `pip install -U mrpt3`"), kind="warn"),
    )

    rng = np.random.default_rng(seed.value)
    _xy = rng.uniform(-15, 15, size=(n_nodes.value, 2))
    _phi = rng.uniform(-np.pi, np.pi, size=n_nodes.value)
    poses = [CPose2D(float(x), float(y), float(p)) for (x, y), p in zip(_xy, _phi)]

    graph = CNetworkOfPoses2D()
    for _i, _p in enumerate(poses):
        graph.setNodePose(_i, _p)
    for _i in range(len(poses)):
        for _j in range(_i + 1, len(poses)):
            if (poses[_j] - poses[_i]).norm() < radius.value:
                graph.insertEdge(_i, _j, poses[_j] - poses[_i])
    return graph, poses


@app.cell(hide_code=True)
def _(mo, n_nodes):
    target = mo.ui.slider(
        start=1, stop=n_nodes.value - 1, step=1, value=min(18, n_nodes.value - 1),
        label="Shortest path from node 0 to node", show_value=True,
    )
    target
    return (target,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Two meanings of "shortest"

    `dijkstra_path(source, target)` returns the list of node IDs of the
    cheapest path. By default each edge costs 1, so it finds the path with
    **fewest edges**. Its optional `edge_weight` argument is a function
    `(from_id, to_id, edge) -> cost`; with `lambda a, b, edge: edge.norm()`
    the cost is the **metric distance**. The C++ class behind both,
    `mrpt::graphs::CDijkstra`, takes the same kind of function.
    """)
    return


@app.function
def metric(from_id, to_id, edge):
    """Edge cost for Dijkstra: the length of the edge, in meters."""
    return edge.norm()


@app.cell
def _(graph, target):
    def shortest_path(edge_weight=None):
        """Node IDs of the cheapest path from node 0 to the target, or None if unreachable."""
        if not graph.getNeighborsOf(0):  # node 0 has no edges
            return None
        try:
            return graph.dijkstra_path(0, target.value, edge_weight=edge_weight)
        except ValueError:  # target not connected to node 0
            return None

    path_hops = shortest_path()
    path_metric = shortest_path(metric)
    return path_hops, path_metric


@app.cell(hide_code=True)
def _(mo, np, path_hops, path_metric, poses, target):
    def length(path):
        return sum((poses[b] - poses[a]).norm() for a, b in zip(path[:-1], path[1:]))

    if path_hops is None:
        _summary = mo.callout(
            mo.md(f"Node {target.value} is not reachable from node 0: increase the link radius."), kind="warn"
        )
    else:
        _summary = mo.md(
            f"""
    | Edge cost | Path from node 0 to node {target.value} | Edges | Length (m) |
    |---|---|---|---|
    | Fewest edges (default) | `{path_hops}` | {len(path_hops) - 1} | {length(path_hops):.1f} |
    | Metric distance (`edge_weight=metric`) | `{path_metric}` | {len(path_metric) - 1} | {length(path_metric):.1f} |

    The straight-line distance between both nodes is {np.hypot(poses[target.value].x - poses[0].x, poses[target.value].y - poses[0].y):.1f} m.
    """
        )
    _summary
    return


@app.cell(hide_code=True)
def _(graph, np, path_hops, path_metric, plt, poses, target):
    _xy = np.array([[p.x, p.y] for p in poses])

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for _k, (_i, _j, _e) in enumerate(graph.getEdges()):
        ax.plot(*_xy[[_i, _j]].T, "-", color="0.75", lw=1, zorder=1, label="Edges" if _k == 0 else None)
    if path_hops is not None:
        ax.plot(*_xy[path_hops].T, "-", color="tab:orange", lw=5, alpha=0.6, zorder=2, label="Fewest edges")
        ax.plot(*_xy[path_metric].T, "-", color="tab:purple", lw=2.5, zorder=3, label="Metric distance")
    ax.plot(*_xy.T, "o", color="tab:blue", ms=6, zorder=4)
    for _i, (_x, _y) in enumerate(_xy):
        ax.annotate(str(_i), (_x, _y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.plot(*_xy[0], "ks", ms=10, zorder=5, label="Source (node 0)")
    ax.plot(*_xy[target.value], "*", color="gold", mec="k", ms=18, zorder=5, label=f"Target (node {target.value})")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The shortest-path tree

    Dijkstra does not compute a single path: it finds the cheapest path from
    the source to **every** reachable node at once. Together, these paths form
    a spanning tree rooted at the source. `dijkstra_tree(source, edge_weight)`
    returns it as a dict `{node: parent}`, and `getNodeDistances(source,
    edge_weight)` the cost of reaching each node. Switch the **edge cost**
    above to see how the tree changes.
    """)
    return


@app.cell
def _(cost, graph):
    _weight = metric if cost.value == "Metric distance" else None
    if graph.getNeighborsOf(0):
        tree = graph.dijkstra_tree(0, edge_weight=_weight)
        distances = graph.getNodeDistances(0, edge_weight=_weight)
    else:
        tree, distances = {}, {0: 0.0}
    return distances, tree


@app.cell(hide_code=True)
def _(cost, distances, graph, np, plt, poses, tree):
    _xy = np.array([[p.x, p.y] for p in poses])

    fig2, ax2 = plt.subplots(figsize=(7, 6.5))
    for _i, _j, _e in graph.getEdges():
        ax2.plot(*_xy[[_i, _j]].T, "-", color="0.85", lw=1, zorder=1)
    for _child, _parent in tree.items():
        ax2.annotate(
            "", xy=_xy[_child], xytext=_xy[_parent], zorder=2,
            arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=1.8, shrinkA=4, shrinkB=6),
        )
    _reached = sorted(distances)
    _unreached = [i for i in range(len(poses)) if i not in distances]
    _sc = ax2.scatter(*_xy[_reached].T, c=[distances[i] for i in _reached], cmap="viridis", s=60, zorder=3)
    if _unreached:
        ax2.plot(*_xy[_unreached].T, "x", color="tab:red", ms=8, zorder=3, label="Unreachable")
        ax2.legend(loc="upper right", fontsize=8)
    for _i, (_x, _y) in enumerate(_xy):
        ax2.annotate(str(_i), (_x, _y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax2.plot(*_xy[0], "ks", ms=10, zorder=4)
    _units = "m" if cost.value == "Metric distance" else "edges"
    fig2.colorbar(_sc, ax=ax2, fraction=0.04, pad=0.02, label=f"Cost from node 0 ({_units})")
    ax2.set_title(f"Shortest-path tree from node 0 ({cost.value.lower()})")
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Traversing the tree

    The C++ example `graphs_dijkstra_example` prints this tree with
    depth-first and breadth-first traversals. With the `{node: parent}` dict,
    both are a few lines of Python:
    """)
    return


@app.cell
def _(mo, tree):
    children = {}
    for _child, _parent in sorted(tree.items()):
        children.setdefault(_parent, []).append(_child)

    def depth_first(node, depth=0):
        """Lines of the subtree of node, indented by depth."""
        lines = ["   " * depth + str(node)]
        for _c in children.get(node, []):
            lines += depth_first(_c, depth + 1)
        return lines

    def breadth_first(root):
        """Node IDs in order of increasing number of tree edges from the root."""
        order, queue = [], [root]
        while queue:
            order.append(queue.pop(0))
            queue += children.get(order[-1], [])
        return order

    mo.hstack(
        [
            mo.md("**Depth-first**\n\n```\n" + "\n".join(depth_first(0)) + "\n```"),
            mo.md("**Breadth-first**\n\n```\n" + " ".join(map(str, breadth_first(0))) + "\n```"),
        ],
        justify="start",
        gap=4,
        align="start",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/dijkstra_shortest_paths.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    The C++ version of this example:
    [graphs_dijkstra_example](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_cpp/graphs_dijkstra_example).
    """)
    return


if __name__ == "__main__":
    app.run()
