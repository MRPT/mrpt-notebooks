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
    # Geodetic coordinates: WGS84, ECEF, ENU and UTM

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to convert GNSS positions between the coordinate
    systems used in outdoor robotics, with the `mrpt.topography` module:

    - `TGeodeticCoords`, `TCoords`: latitude, longitude and height on the WGS84 ellipsoid.
    - `geodeticToGeocentric_WGS84()`, `geocentricToGeodetic()`: Earth-Centered, Earth-Fixed (ECEF) coordinates.
    - `geodeticToENU_WGS84()`, `ENUToGeodetic_WGS84()`, `ENU_axes_from_WGS84()`: local East-North-Up (ENU) frames.
    - `geodeticToUTM()`, `UTMToGeodetic()`: the UTM map projection.

    **Pick a place and a survey loop size**: everything below is recomputed. Try
    Greenwich or the equator to see UTM break, and large loops to see the Earth
    curvature in ENU.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mrpt.math import TPoint3D
    from mrpt.topography import (
        ENU_axes_from_WGS84,
        ENUToGeodetic_WGS84,
        TGeodeticCoords,
        UTMToGeodetic,
        geocentricToGeodetic,
        geodeticToENU_WGS84,
        geodeticToGeocentric_WGS84,
        geodeticToUTM,
    )
    return (
        ENUToGeodetic_WGS84,
        ENU_axes_from_WGS84,
        TGeodeticCoords,
        TPoint3D,
        UTMToGeodetic,
        geocentricToGeodetic,
        geodeticToENU_WGS84,
        geodeticToGeocentric_WGS84,
        geodeticToUTM,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    # Latitude (deg), longitude (deg), ellipsoidal height (m):
    places = {
        "Almería, Spain": (36.8340, -2.4637, 20.0),
        "Greenwich, UK (on a UTM zone boundary)": (51.4778, -0.0015, 46.0),
        "Mitad del Mundo, Ecuador (on the equator)": (-0.0022, -78.4558, 2480.0),
        "Sydney, Australia (southern hemisphere)": (-33.8568, 151.2153, 25.0),
        "Tromsø, Norway (high latitude)": (69.6492, 18.9553, 30.0),
    }
    place = mo.ui.dropdown(options=list(places), value="Almería, Spain", label="Reference point")
    loop_km = mo.ui.slider(
        steps=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], value=10, label="Survey loop radius (km)", show_value=True
    )
    mo.vstack([mo.md("### Parameters"), mo.hstack([place, loop_km], justify="start", gap=2)])
    return loop_km, place, places


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One point, four coordinate systems

    - **Geodetic (WGS84)**: latitude, longitude and height above the WGS84
      ellipsoid, as given by any GNSS receiver. Angles are not metric: a degree of
      longitude shrinks with the latitude.
    - **ECEF**: Cartesian (x, y, z) in meters, with the origin at the Earth
      center, z toward the North pole and x toward longitude 0°. Global and
      metric, but its axes have nothing to do with the local horizontal.
    - **ENU**: Cartesian frame tangent to the ellipsoid at a reference point:
      x East, y North, z Up. It is just ECEF rotated and translated
      (`ENU_axes_from_WGS84()` gives that rigid transform).
    - **UTM**: a map projection. The Earth is split into 60 zones of 6° of
      longitude, each one projected onto a plane (easting, northing). The
      letter is the latitude band (C to X, from South to North), not the
      hemisphere: bands N and above are in the northern one.
    """)
    return


@app.cell
def _(
    ENU_axes_from_WGS84,
    TGeodeticCoords,
    geodeticToGeocentric_WGS84,
    geodeticToUTM,
    place,
    places,
):
    ref = TGeodeticCoords(*places[place.value])
    ref_ecef = geodeticToGeocentric_WGS84(ref)
    ref_utm, ref_zone, ref_band = geodeticToUTM(ref)
    enu_frame = ENU_axes_from_WGS84(ref)  # a TPose3D: the ENU frame, in ECEF coordinates
    return enu_frame, ref, ref_ecef, ref_band, ref_utm, ref_zone


@app.cell(hide_code=True)
def _(enu_frame, mo, np, plt, ref, ref_band, ref_ecef, ref_utm, ref_zone):
    def dms(c, pos, neg):
        d, m, s = c.getDegMinSec()  # TCoords: degrees, minutes, seconds
        return f"{abs(d)}° {abs(m)}' {abs(s):.2f}\" {pos if c.decimal_value >= 0 else neg}"

    _table = mo.md(
        f"""
    | System | Coordinates |
    |---|---|
    | Geodetic | lat {ref.lat.decimal_value:.6f}°, lon {ref.lon.decimal_value:.6f}°, h {ref.height:.1f} m |
    | Geodetic (DMS) | {dms(ref.lat, "N", "S")}, {dms(ref.lon, "E", "W")} |
    | ECEF | x {ref_ecef.x:,.2f} m, y {ref_ecef.y:,.2f} m, z {ref_ecef.z:,.2f} m |
    | UTM | zone **{ref_zone}{ref_band}**, easting {ref_utm.x:,.2f} m, northing {ref_utm.y:,.2f} m |
    | ENU | (0, 0, 0): this point is the origin of the local frame |
    """
    )

    # A sketch of the Earth (unit sphere), with the ECEF axes and the local ENU frame:
    _fig = plt.figure(figsize=(4.6, 4.6))
    _ax = _fig.add_subplot(projection="3d")
    _u, _v = np.meshgrid(np.linspace(-np.pi, np.pi, 37), np.linspace(-np.pi / 2, np.pi / 2, 19))
    _ax.plot_wireframe(np.cos(_v) * np.cos(_u), np.cos(_v) * np.sin(_u), np.sin(_v), color="0.8", lw=0.5)
    for _i, _name in enumerate("xyz"):
        _ax.quiver(0, 0, 0, *(1.6 * np.eye(3)[_i]), color="0.3", lw=1.5, arrow_length_ratio=0.08)
        _ax.text(*(1.75 * np.eye(3)[_i]), f"ECEF {_name}", color="0.3", fontsize=8)
    _p = np.array([ref_ecef.x, ref_ecef.y, ref_ecef.z]) / 6371e3
    _ax.scatter(*_p, color="k", s=20)
    _R = np.array(enu_frame.getRotationMatrix())  # columns: E, N, U axes in ECEF
    for _i, (_name, _color) in enumerate(zip("ENU", ["#e34948", "#008300", "#2a78d6"])):
        _ax.quiver(*_p, *(0.6 * _R[:, _i]), color=_color, lw=2.5, arrow_length_ratio=0.15)
        _ax.text(*(_p + 0.8 * _R[:, _i]), _name, color=_color, fontsize=10, weight="bold")
    _ax.view_init(elev=np.copysign(20, ref.lat.decimal_value), azim=ref.lon.decimal_value + 45)
    _ax.set_box_aspect((1, 1, 1))
    _ax.set_axis_off()
    _ax.set_title("ECEF axes and the local ENU frame", fontsize=10)
    _fig.tight_layout()

    mo.hstack([_table, _fig], justify="space-between", align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A GNSS survey loop

    Now we simulate a surveyor (or a robot) driving a closed loop around the
    reference point, logging 360 GNSS fixes (latitude, longitude, and the same
    ellipsoidal height as the reference). Each fix is converted to ENU, ECEF and
    UTM with MRPT, and then back to geodetic to check the round trips.
    """)
    return


@app.cell
def _(
    ENUToGeodetic_WGS84,
    TGeodeticCoords,
    TPoint3D,
    UTMToGeodetic,
    enu_frame,
    geocentricToGeodetic,
    geodeticToENU_WGS84,
    geodeticToGeocentric_WGS84,
    geodeticToUTM,
    loop_km,
    np,
    ref,
):
    # The survey: a wobbly loop around the reference. Spherical-Earth offsets are
    # good enough to generate test data in degrees:
    _th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    _r = 1e3 * loop_km.value * (1 + 0.2 * np.sin(3 * _th) + 0.1 * np.cos(5 * _th))
    _lat0, _lon0 = ref.lat.decimal_value, ref.lon.decimal_value
    lats = _lat0 + np.degrees(_r * np.sin(_th) / 6371e3)
    lons = _lon0 + np.degrees(_r * np.cos(_th) / (6371e3 * np.cos(np.radians(_lat0))))
    fixes = [TGeodeticCoords(la, lo, ref.height) for la, lo in zip(lats, lons)]

    def xyz(p):
        return [p.x, p.y, p.z]

    enu = np.array([xyz(geodeticToENU_WGS84(g, ref)) for g in fixes])
    ecef = np.array([xyz(geodeticToGeocentric_WGS84(g)) for g in fixes])
    _utm = [geodeticToUTM(g) for g in fixes]  # (TPoint3D, zone, band) tuples
    utm = np.array([xyz(u) for u, _, _ in _utm])
    utm_zones = [f"{z}{b}" for _, z, b in _utm]

    # Round trips back to geodetic, worst error in meters (via ENU of the result):
    def roundtrip_error(back):
        return max(np.linalg.norm(xyz(geodeticToENU_WGS84(b, g))) for b, g in zip(back, fixes))

    err_enu = roundtrip_error([ENUToGeodetic_WGS84(TPoint3D(*p), ref) for p in enu])
    err_ecef = roundtrip_error([geocentricToGeodetic(TPoint3D(*p)) for p in ecef])
    err_utm = roundtrip_error([UTMToGeodetic(u, z, band=b) for u, z, b in _utm])

    # ENU is a rigid transform of ECEF, given by the pose of the ENU frame (a TPose3D):
    _enu2 = np.array([xyz(enu_frame.inverseComposePoint(TPoint3D(*p))) for p in ecef])
    err_frame = np.abs(_enu2 - enu).max()
    return ecef, enu, err_ecef, err_enu, err_frame, err_utm, utm, utm_zones


@app.cell
def _(ENUToGeodetic_WGS84, TPoint3D, geodeticToUTM, loop_km, np, ref, ref_utm):
    # Where does true North point on the UTM grid? Project a point 1 m North of the reference:
    _north = geodeticToUTM(ENUToGeodetic_WGS84(TPoint3D(0.0, 1.0, 0.0), ref))[0]
    _dir = np.array([_north.x - ref_utm.x, _north.y - ref_utm.y])
    convergence = np.degrees(np.arctan2(_dir[0], _dir[1]))
    north_arrow = 0.5e3 * loop_km.value * _dir / np.linalg.norm(_dir)  # for the plot
    return convergence, north_arrow


@app.cell(hide_code=True)
def _(convergence, ecef, enu, err_ecef, err_enu, err_frame, err_utm, mo, np, ref_band, ref_zone, utm, utm_zones):
    def loop_length(xy):
        return np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1).sum()

    _len_true = loop_length(ecef)
    _len_enu = loop_length(enu)
    _len_utm = loop_length(utm[:, :2])
    _zones = sorted(set(utm_zones))
    _zone_msg = (
        mo.callout(
            mo.md(
                f"The loop crosses **UTM zones {', '.join(_zones)}**: UTM coordinates jump "
                "between them, so a single UTM frame cannot hold this survey."
            ),
            kind="danger",
        )
        if len(_zones) > 1
        else mo.callout(mo.md(f"The whole loop lies in UTM zone **{ref_zone}{ref_band}**."), kind="success")
    )
    mo.vstack(
        [
            _zone_msg,
            mo.md(
                f"""
    | Loop length computed in | Length (m) | Difference (m) |
    |---|---|---|
    | ECEF (true 3D distances) | {_len_true:,.3f} | |
    | ENU (x, y, z) | {_len_enu:,.3f} | {_len_enu - _len_true:+.3f} |
    | UTM (easting, northing) | {_len_utm:,.3f} | {_len_utm - _len_true:+,.3f} |

    ENU keeps distances exactly, because it is a rigid transformation of ECEF:
    transforming the ECEF points with the pose from `ENU_axes_from_WGS84()`
    gives the same ENU coordinates (max. difference {err_frame * 1e3:.1e} mm).
    UTM does not: its scale factor goes from 0.9996 at the zone central meridian
    to about 1.001 at the zone edges, and it measures on the ellipsoid surface,
    not at the actual height. Its grid North also differs from true North by
    **{convergence:+.3f}°** here (meridian convergence).

    Worst round-trip errors back to geodetic: ENU {err_enu * 1e3:.1e} mm,
    ECEF {err_ecef * 1e3:.1e} mm, UTM {err_utm * 1e3:.1e} mm.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(enu, loop_km, north_arrow, np, plt, ref_utm, utm, utm_zones):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.9))

    # ENU map, colored by the "up" coordinate:
    _sc = ax1.scatter(enu[:, 0] / 1e3, enu[:, 1] / 1e3, c=enu[:, 2], s=8, cmap="viridis")
    ax1.plot(0, 0, "k+", ms=12, mew=2)
    ax1.set_title("ENU frame")
    ax1.set_xlabel("East (km)")
    ax1.set_ylabel("North (km)")
    fig.colorbar(_sc, ax=ax1, label="Up (m)", shrink=0.85)

    # UTM map, relative to the reference point, one color per zone:
    for _zone, _color in zip(sorted(set(utm_zones)), ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]):
        _in = np.array(utm_zones) == _zone
        ax2.scatter(
            (utm[_in, 0] - ref_utm.x) / 1e3, (utm[_in, 1] - ref_utm.y) / 1e3, s=8, color=_color, label=f"zone {_zone}"
        )
    ax2.plot(0, 0, "k+", ms=12, mew=2)
    if len(set(utm_zones)) == 1:
        ax2.annotate("", xy=north_arrow / 1e3, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#e34948", lw=2))
        ax2.text(*(1.15 * north_arrow / 1e3), "true North", color="#e34948", ha="center", va="bottom")
    ax2.set_title("UTM, relative to the reference")
    ax2.set_xlabel("Easting (km)")
    ax2.set_ylabel("Northing (km)")
    ax2.legend(loc="lower right", fontsize=8)

    # Earth curvature: all fixes have the same height, but "up" drops with the distance:
    _d = np.hypot(enu[:, 0], enu[:, 1])
    _dd = np.linspace(0, _d.max(), 100)
    ax3.scatter(_d / 1e3, enu[:, 2], s=25, color="#2a78d6", label="ENU up of each fix")
    ax3.plot(_dd / 1e3, -(_dd**2) / (2 * 6371e3), "k--", lw=1.2, label="$-d^2 / (2 R_{Earth})$")
    ax3.set_title("Same height, but ENU up decreases")
    ax3.set_xlabel("Horizontal distance from the reference (km)")
    ax3.set_ylabel("Up (m)")
    ax3.legend(loc="lower left", fontsize=8)

    for _ax in (ax1, ax2):
        _ax.set_aspect("equal", adjustable="datalim")
    for _ax in (ax1, ax2, ax3):
        _ax.grid(alpha=0.3)
    fig.suptitle(f"Survey loop of {loop_km.value:g} km radius", fontsize=11)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why robots use local ENU frames

    Robot navigation, SLAM and sensor fusion need a metric, Cartesian frame
    where x and y are horizontal and z is vertical: ENU is exactly that around
    its origin, with no projection distortion and no zone boundaries. Geodetic
    angles are not metric, ECEF has no notion of "up", and UTM distorts
    distances and breaks at zone and hemisphere boundaries.

    ENU is a plane tangent to a curved Earth, though: points at a constant height
    drop below it as $d^2 / (2 R)$, about 8 cm at 1 km and 8 m at 10 km
    (right plot). So ENU frames are kept *local*: a few kilometers around a
    reference point, typically the first GNSS fix of the mission. Larger areas
    are split into several local frames.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/geodetic_coordinates.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
