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
    # 3D metric maps from a point cloud

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to build five different 3D map representations from
    the same simulated 3D LiDAR scan, and compares them. All maps live in
    `mrpt.maps` and share the `CMetricMap` interface:

    - `CSimplePointsMap`: the raw point cloud.
    - `CVoxelMap`: a sparse voxel map with occupancy probabilities.
    - `COctoMap`: an octree of occupancy probabilities (the OctoMap library).
    - `COccupancyGridMap3D`: a dense 3D occupancy grid.
    - `CHeightGridMap2D`: a 2.5D elevation map (one height per cell).

    The scan is wrapped in a `CObservationPointCloud` (`mrpt.maps`) and inserted
    into every map with the same call, `insertObservation()`, from a `CPose3D`
    sensor pose (`mrpt.poses`). **Move the sliders**: change the map resolution,
    pick which map to see in 3D, and aim a probe ray to query the maps.
    """)
    return


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go

    from mrpt.maps import (
        CHeightGridMap2D,
        CObservationPointCloud,
        COccupancyGridMap3D,
        COctoMap,
        CSimplePointsMap,
        CVoxelMap,
    )
    from mrpt.math import TPoint3D
    from mrpt.poses import CPose3D
    return (
        CHeightGridMap2D,
        CObservationPointCloud,
        COccupancyGridMap3D,
        COctoMap,
        CPose3D,
        CSimplePointsMap,
        CVoxelMap,
        TPoint3D,
        go,
        mo,
        np,
        plt,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. A simulated 3D LiDAR scan

    The world is a walled yard (16 x 12 m) with a table, a cabinet, stacked
    crates, a ball and a tree. A 32-ring rotating LiDAR, 1.3 m above the ground,
    casts 32,768 rays (elevations from -30° to +10°) against it. Ray casting is
    done with NumPy, and the resulting points (in the sensor frame) go into a
    `CSimplePointsMap`.
    """)
    return


@app.cell
def _(np):
    def box(cx, cy, z0, sx, sy, h):
        """Axis-aligned box [xmin, ymin, zmin, xmax, ymax, zmax]."""
        return [cx - sx / 2, cy - sy / 2, z0, cx + sx / 2, cy + sy / 2, z0 + h]

    boxes = np.array(
        [
            box(0.0, 6.1, 0.0, 16.4, 0.2, 2.5),  # walls
            box(0.0, -6.1, 0.0, 16.4, 0.2, 2.5),
            box(8.1, 0.0, 0.0, 0.2, 12.4, 2.5),
            box(-8.1, 0.0, 0.0, 0.2, 12.4, 2.5),
            box(2.5, 2.0, 0.72, 1.8, 0.9, 0.06),  # table top and legs
            *[box(2.5 + sx * 0.8, 2.0 + sy * 0.35, 0.0, 0.06, 0.06, 0.72) for sx in (-1, 1) for sy in (-1, 1)],
            box(-5.0, 4.5, 0.0, 1.2, 0.6, 1.9),  # cabinet
            box(-3.0, -3.5, 0.0, 1.0, 1.0, 1.0),  # crates
            box(-3.0, -3.5, 1.0, 0.6, 0.6, 0.6),
            box(5.5, -3.0, 0.0, 0.3, 0.3, 1.6),  # tree trunk
        ]
    )
    spheres = np.array([[5.5, -3.0, 2.3, 1.0], [0.5, -2.0, 0.4, 0.4]])  # tree crown, ball: [x, y, z, radius]

    def lidar_scan(sensor, rng, n_rings=32, n_az=1024, max_range=25.0, sigma=0.01):
        """Ray-casts a rotating LiDAR at `sensor` (x, y, z). Returns (N, 3) points in the sensor frame."""
        el = np.radians(np.linspace(-30, 10, n_rings))
        az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
        el, az = np.meshgrid(el, az)
        d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1).reshape(-1, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_ground = np.where(d[:, 2] < 0, -sensor[2] / d[:, 2], np.inf)
            # Ray-box intersection (slab method), for all rays and boxes at once:
            t1 = (boxes[None, :, :3] - sensor) / d[:, None, :]
            t2 = (boxes[None, :, 3:] - sensor) / d[:, None, :]
        t_in = np.minimum(t1, t2).max(axis=-1)
        t_out = np.maximum(t1, t2).min(axis=-1)
        t_box = np.where((t_out >= t_in) & (t_in > 0), t_in, np.inf).min(axis=-1)
        # Ray-sphere intersection:
        oc = spheres[:, :3] - sensor
        b = d @ oc.T
        disc = b**2 - (oc**2).sum(axis=1) + spheres[:, 3] ** 2
        with np.errstate(invalid="ignore"):
            t_sph = np.where(disc > 0, b - np.sqrt(disc), np.inf)
        t_sph = np.where(t_sph > 0, t_sph, np.inf).min(axis=-1)
        t = np.minimum(np.minimum(t_ground, t_box), t_sph)
        ok = t < max_range  # rays to the sky return nothing
        r = t[ok] + rng.normal(0.0, sigma, size=ok.sum())
        return (d[ok] * r[:, None]).astype(np.float32)
    return (lidar_scan,)


@app.cell
def _(CObservationPointCloud, CPose3D, CSimplePointsMap, lidar_scan, mo, np):
    sensor_xyz = np.array([-1.0, 0.5, 1.3])
    sensor_pose = CPose3D(*sensor_xyz, 0.0, 0.0, 0.0)  # x, y, z, yaw, pitch, roll

    scan_local = lidar_scan(sensor_xyz, np.random.default_rng(1))
    obs = CObservationPointCloud()
    obs.pointcloud = CSimplePointsMap()
    obs.pointcloud.setPointsFromNumpy(scan_local)
    scan_world = scan_local + sensor_xyz.astype(np.float32)  # the sensor is not rotated

    mo.md(f"Simulated scan: **{len(obs.pointcloud):,} points** in a `{type(obs).__name__}`.")
    return obs, scan_world, sensor_pose, sensor_xyz


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Five map representations

    - **Point cloud**: keeps every measurement, no resolution, no free space.
    - **Occupancy maps** (`CVoxelMap`, `COctoMap`, `COccupancyGridMap3D`): each
      ray marks the voxels it crosses as *free* and its end voxel as *occupied*,
      fusing evidence as log-odds probabilities. Voxels never crossed by a ray stay
      *unknown*. They differ in how voxels are stored: a sparse block grid, an
      octree, or a dense array that allocates every voxel of the bounding box.
    - **Height map** (`CHeightGridMap2D`): stores one mean height per (x, y)
      cell. Compact and good for terrain, but it cannot represent overhangs (the
      space under the table or the tree crown) and averages vertical surfaces.

    The occupancy maps have no bulk export in Python, except
    `CVoxelMap.getOccupiedVoxels()`: for the other two, the 3D view queries their
    occupancy at the voxels that contain a measured point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    resolution = mo.ui.slider(start=0.1, stop=0.5, step=0.05, value=0.2, label="Map resolution (m)", show_value=True)
    view = mo.ui.dropdown(
        options=[
            "CSimplePointsMap",
            "CVoxelMap",
            "COctoMap",
            "COccupancyGridMap3D",
            "CHeightGridMap2D",
        ],
        value="CVoxelMap",
        label="Show in 3D",
    )
    probe_az = mo.ui.slider(start=-180, stop=180, step=1, value=-117, label="Probe ray: azimuth (deg)", show_value=True)
    probe_el = mo.ui.slider(start=-30, stop=10, step=0.5, value=0, label="elevation (deg)", show_value=True)

    mo.vstack(
        [
            mo.md("### Parameters"),
            mo.hstack([resolution, view], justify="start", gap=2),
            mo.hstack([probe_az, probe_el], justify="start", gap=2),
        ]
    )
    return probe_az, probe_el, resolution, view


@app.cell
def _(
    CHeightGridMap2D,
    COccupancyGridMap3D,
    COctoMap,
    CSimplePointsMap,
    CVoxelMap,
    TPoint3D,
    np,
    obs,
    resolution,
    scan_world,
    sensor_pose,
    time,
):
    res = resolution.value
    # Map limits (for the dense maps), aligned to the voxel lattice:
    lo = np.floor(scan_world.min(axis=0) / res - 1) * res
    hi = np.ceil(scan_world.max(axis=0) / res + 1) * res

    maps = {
        "CSimplePointsMap": CSimplePointsMap(),
        "CVoxelMap": CVoxelMap(res),
        "COctoMap": COctoMap(res),
        "COccupancyGridMap3D": COccupancyGridMap3D(TPoint3D(*lo), TPoint3D(*hi), res),
        "CHeightGridMap2D": CHeightGridMap2D(lo[0], hi[0], lo[1], hi[1], res),
    }
    insert_ms = {}
    for _name, _m in maps.items():
        _t0 = time.perf_counter()
        _m.insertObservation(obs, sensor_pose)  # the same call for all map types
        insert_ms[_name] = 1e3 * (time.perf_counter() - _t0)
    return insert_ms, maps, res


@app.cell
def _(maps, np, res, scan_world):
    OCCUPIED = 0.6  # probability threshold, the default of CVoxelMap.getOccupiedVoxels()

    # Centers of the voxels that contain at least one measured point:
    candidates = (np.unique(np.floor(scan_world / res), axis=0) + 0.5) * res

    def occupancy(name, p):
        """Occupancy probability at a point, or NaN if unknown."""
        m = maps[name]
        if name == "COccupancyGridMap3D":
            return 1.0 - m.getFreenessByPos(*p)  # this class stores "freeness"
        prob = m.getPointOccupancy(*p)  # None if never observed
        return np.nan if prob is None else prob

    occupied = {
        "CSimplePointsMap": maps["CSimplePointsMap"].getPointsAsNumpy(),
        "CVoxelMap": maps["CVoxelMap"].getOccupiedVoxels().getPointsAsNumpy(),
    }
    for _name in ["COctoMap", "COccupancyGridMap3D"]:
        _p = np.array([occupancy(_name, c) for c in candidates])
        occupied[_name] = candidates[_p > OCCUPIED]
    return occupancy, occupied


@app.cell(hide_code=True)
def _(insert_ms, maps, mo, occupied, res):
    _g3 = maps["COccupancyGridMap3D"]
    _hm = maps["CHeightGridMap2D"]
    _n3 = _g3.getSizeX() * _g3.getSizeY() * _g3.getSizeZ()
    _n2 = _hm.getSizeX() * _hm.getSizeY()
    _npts = len(maps["CSimplePointsMap"])
    mo.md(
        f"""
    | Map class | What is stored | Size at {res:.2f} m | Insertion time |
    |---|---|---|---|
    | `CSimplePointsMap` | x, y, z of each point (float32) | {_npts:,} points ({12 * _npts / 1e6:.2f} MB) | {insert_ms["CSimplePointsMap"]:.1f} ms |
    | `CVoxelMap` | blocks of observed voxels only (sparse) | {len(occupied["CVoxelMap"]):,} occupied voxels | {insert_ms["CVoxelMap"]:.1f} ms |
    | `COctoMap` | octree nodes (free, occupied, inner) | {maps["COctoMap"].size():,} nodes, {len(occupied["COctoMap"]):,} occupied voxels | {insert_ms["COctoMap"]:.1f} ms |
    | `COccupancyGridMap3D` | every voxel of the box (1 byte each) | {_g3.getSizeX()} x {_g3.getSizeY()} x {_g3.getSizeZ()} = {_n3:,} voxels ({_n3 / 1e6:.2f} MB), {len(occupied["COccupancyGridMap3D"]):,} occupied | {insert_ms["COccupancyGridMap3D"]:.1f} ms |
    | `CHeightGridMap2D` | one height per (x, y) cell | {_hm.getSizeX()} x {_hm.getSizeY()} = {_n2:,} cells, {_hm.countObservedCells():,} observed | {insert_ms["CHeightGridMap2D"]:.1f} ms |

    The dense grid grows with the volume of its bounding box (x8 when halving the
    voxel size), observed or not, while the sparse maps only store what the rays
    touched. Ray tracing makes the occupancy maps slower to build than a point
    cloud, in exchange for knowing which space is free.
    """
    )
    return


@app.cell
def _(np, probe_az, probe_el, sensor_xyz):
    _az, _el = np.radians(probe_az.value), np.radians(probe_el.value)
    probe_dir = np.array([np.cos(_el) * np.cos(_az), np.cos(_el) * np.sin(_az), np.sin(_el)])
    # Up to 12 m, or until 0.5 m below the ground:
    probe_len = min(12.0, (sensor_xyz[2] + 0.5) / max(-probe_dir[2], 1e-6))
    return probe_dir, probe_len


@app.cell(hide_code=True)
def _(go, maps, np, occupied, probe_dir, probe_len, res, sensor_xyz, view):
    fig3d = go.Figure()
    if view.value == "CHeightGridMap2D":
        _hm = maps["CHeightGridMap2D"]
        _r = _hm.getResolution()
        _h = _hm.getAsNumpy()  # rows: y, columns: x (NaN = unobserved)
        fig3d.add_trace(
            go.Surface(
                x=_hm.getXMin() + _r * (np.arange(_h.shape[1]) + 0.5),
                y=_hm.getYMin() + _r * (np.arange(_h.shape[0]) + 0.5),
                z=_h,
                colorscale="Viridis",
                cmin=0.0,
                cmax=2.5,
                colorbar=dict(title="height (m)", len=0.6),
            )
        )
    else:
        _xyz = occupied[view.value]
        if len(_xyz) > 20000:  # keep the page light
            _xyz = _xyz[np.random.default_rng(0).choice(len(_xyz), 20000, replace=False)]
        _size = 1.5 if view.value == "CSimplePointsMap" else 1.5 + 12 * res
        fig3d.add_trace(
            go.Scatter3d(
                x=_xyz[:, 0],
                y=_xyz[:, 1],
                z=_xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=_size,
                    symbol="square",
                    color=_xyz[:, 2],
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=2.5,
                    colorbar=dict(title="z (m)", len=0.6),
                ),
                name=view.value,
            )
        )
    _end = sensor_xyz + probe_len * probe_dir
    fig3d.add_trace(
        go.Scatter3d(
            x=[sensor_xyz[0], _end[0]],
            y=[sensor_xyz[1], _end[1]],
            z=[sensor_xyz[2], _end[2]],
            mode="lines+markers",
            line=dict(color="#e34948", width=6),
            marker=dict(size=[6, 0], color="#e34948"),
            name="Probe ray",
        )
    )
    fig3d.update_layout(
        title=f"{view.value} at {res:.2f} m",
        height=560,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        scene=dict(
            aspectmode="data",
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            camera=dict(eye=dict(x=-0.9, y=-1.3, z=1.1)),
        ),
    )
    fig3d
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Querying the maps along a ray

    The red probe ray starts at the sensor. Below, the occupancy maps are queried
    along it (`getPointOccupancy()`, `getFreenessByPos()`). By default it hits the
    stacked crates: the probability is low in free space, high at the crate, and
    unknown behind it (the dense grid cannot tell *unknown* from p = 0.5).
    `COctoMap` updates each voxel once per scan (hits win), hence its flat
    0.4 / 0.7 values, while the other two maps fuse every ray.

    The bottom plot compares the ray height with the height map (`getHeight()`)
    along the same path. Each cell holds the mean height of its points, so the
    crates (1.6 m tall) look lower than they are, and their vertical front face
    becomes a ramp.
    """)
    return


@app.cell(hide_code=True)
def _(maps, np, occupancy, plt, probe_dir, probe_len, res, sensor_xyz):
    _dist = np.arange(0.0, probe_len, res / 4)
    _ray = sensor_xyz + _dist[:, None] * probe_dir

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, height_ratios=[3, 2])
    for _name, _color in [("CVoxelMap", "#2a78d6"), ("COctoMap", "#eb6834"), ("COccupancyGridMap3D", "#1baf7a")]:
        ax1.plot(_dist, [occupancy(_name, p) for p in _ray], color=_color, lw=2, label=_name)
    ax1.axhline(0.5, color="0.5", lw=1, ls=":")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_ylabel("p(occupied)")
    ax1.set_title("Occupancy along the probe ray (gaps: unknown)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    _h = [maps["CHeightGridMap2D"].getHeight(x, y) for x, y, _ in _ray]
    ax2.plot(_dist, [np.nan if v is None else v for v in _h], color="#6250d6", lw=2, label="CHeightGridMap2D height")
    ax2.plot(_dist, _ray[:, 2], color="#e34948", lw=2, ls="--", label="Probe ray height")
    ax2.set_xlabel("Distance from the sensor (m)")
    ax2.set_ylabel("z (m)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/point_cloud_maps.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
