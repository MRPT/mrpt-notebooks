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
    # Camera models and lens distortion

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to explore pinhole camera models with lens distortion,
    and their effect on images and on depth-camera point clouds:

    - `mrpt.img`: camera intrinsics and distortion (`TCamera`, `DistortionModel`),
      images (`CImage`) and colormaps (`colormap()`).
    - `mrpt.obs`: depth images (`CObservation3DRangeScan`), turned into 3D points
      by `unprojectInto()`, which undistorts every pixel with the `TCamera` model.

    A camera inside a corridor with checkered walls takes an image and a depth
    image. **Move the sliders**: change the distortion model, the focal length
    and the distortion coefficients, and see what happens to the image and to the
    3D point cloud when the distortion is ignored.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mrpt.img import CImage, DistortionModel, TCamera, TColor, TColormap, TPixelCoord, colormap
    from mrpt.obs import CObservation3DRangeScan
    return (
        CImage,
        CObservation3DRangeScan,
        DistortionModel,
        TCamera,
        TColor,
        TColormap,
        TPixelCoord,
        colormap,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The camera model

    A 3D point $(x, y, z)$ in the camera frame ($z$ forward, $y$ down) is first
    projected onto the normalized plane, $(x/z,\ y/z)$, then **distorted** by the
    lens model, and finally scaled by the focal length and shifted by the
    principal point: $u = f_x x_d + c_x$, $v = f_y y_d + c_y$. MRPT supports
    three distortion models:

    - `none`: an ideal pinhole camera (or an already rectified image).
    - `plumb_bob`: the radial-tangential model also used by OpenCV and ROS.
      Radial coefficients $k_1, k_2$ bend straight lines into barrels (negative
      $k_1$) or pincushions (positive $k_1$); $p_1, p_2$ model a lens not parallel to the sensor.
    - `kannala_brandt`: a fisheye model, a polynomial in the angle $\theta$
      between the ray and the optical axis. With all its coefficients at zero it
      is an *equidistant* fisheye, $r_d = \theta$, with a much wider field of
      view than a pinhole camera of the same focal length.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Shown in the next cell, together with the sliders that depend on it:
    model = mo.ui.dropdown(options=["none", "plumb_bob", "kannala_brandt"], value="plumb_bob", label="Distortion model")
    return (model,)


@app.cell(hide_code=True)
def _(mo, model):
    # Slider ranges depend on the model, keeping the lens model valid over the whole image:
    if model.value == "plumb_bob":
        focal = mo.ui.slider(start=320, stop=640, step=10, value=320, label="Focal length (px)", show_value=True)
        k1 = mo.ui.slider(start=-0.25, stop=0.25, step=0.01, value=-0.2, label="k1", show_value=True)
        k2 = mo.ui.slider(start=0.0, stop=0.1, step=0.005, value=0.05, label="k2", show_value=True)
    elif model.value == "kannala_brandt":
        focal = mo.ui.slider(start=180, stop=400, step=10, value=180, label="Focal length (px)", show_value=True)
        k1 = mo.ui.slider(start=-0.05, stop=0.1, step=0.005, value=0.0, label="k1", show_value=True)
        k2 = mo.ui.slider(start=-0.01, stop=0.02, step=0.001, value=0.0, label="k2", show_value=True)
    else:
        focal = mo.ui.slider(start=100, stop=640, step=10, value=200, label="Focal length (px)", show_value=True)
        k1 = k2 = None
    p1 = mo.ui.slider(start=-0.01, stop=0.01, step=0.001, value=0.0, label="p1", show_value=True)
    p2 = mo.ui.slider(start=-0.01, stop=0.01, step=0.001, value=0.0, label="p2", show_value=True)

    _rows = [mo.md("### Parameters"), mo.hstack([model, focal], justify="start", gap=2)]
    if model.value != "none":
        _rows.append(mo.hstack([k1, k2], justify="start", gap=2))
    if model.value == "plumb_bob":
        _rows.append(mo.hstack([p1, p2], justify="start", gap=2))
    mo.vstack(_rows)
    return focal, k1, k2, p1, p2


@app.cell
def _(DistortionModel, TCamera, focal, k1, k2, model, p1, p2):
    W, H = 320, 240  # image size (px)

    def make_camera(model_name):
        cam = TCamera()
        cam.ncols, cam.nrows = W, H
        cam.fx = cam.fy = focal.value
        cam.cx, cam.cy = (W - 1) / 2, (H - 1) / 2
        cam.distortion = getattr(DistortionModel, model_name)
        if model_name == "plumb_bob":
            cam.dist = [k1.value, k2.value, p1.value, p2.value, 0, 0, 0, 0]  # k1 k2 p1 p2 k3 k4 k5 k6
        elif model_name == "kannala_brandt":
            cam.dist = [k1.value, k2.value, 0, 0, 0, 0, 0, 0]  # k1 k2 - - k3 k4 - -
        return cam

    camera = make_camera(model.value)
    pinhole = make_camera("none")  # the same camera, with its distortion ignored
    return H, W, camera, pinhole


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. What the camera sees

    `CObservation3DRangeScan.unprojectInto()` converts each pixel of a depth
    image into a 3D point, undistorting the pixel with the `TCamera` model. A
    depth image of 1 m everywhere thus gives the **viewing ray of every pixel**.
    We cast those rays against the corridor (NumPy) to render the image and the
    depth image the camera would capture. Both are stored in `CImage` objects,
    annotated with MRPT drawing functions, and the depth is colored with an MRPT
    colormap.
    """)
    return


@app.cell
def _(CObservation3DRangeScan, H, W, camera, np):
    def unproject(cam, depth):
        """Depth image (H, W) in meters -> (H * W, 3) points in the sensor frame (x forward, y left, z up)."""
        obs = CObservation3DRangeScan()
        obs.cameraParams = cam
        obs.range_is_depth = True  # pixels hold depth (x), not the distance along the ray
        obs.maxRange = 100.0
        obs.setRangeImageFromNumpy(depth)
        obs.unprojectInto()
        return obs.getPoints3DAsNumpy()

    # One point at depth 1 m along the ray of each pixel: rays[v, u] = (1, y, z)
    rays = unproject(camera, np.ones((H, W), np.float32)).reshape(H, W, 3)
    fov_h = np.degrees(np.arctan(rays[H // 2, 0, 1]) - np.arctan(rays[H // 2, -1, 1]))
    return fov_h, rays, unproject


@app.cell
def _(np, rays):
    # A corridor: (axis, coordinate, color) of each face, in the sensor frame (x forward, y left, z up).
    CORRIDOR = [
        (0, 6.0, (225, 225, 225)),  # end wall
        (1, 1.0, (235, 104, 52)),  # left wall
        (1, -1.0, (27, 175, 122)),  # right wall
        (2, -0.8, (42, 120, 214)),  # floor
        (2, 0.9, (237, 161, 0)),  # ceiling
    ]

    # Ray casting: the closest face along each ray.
    depth = np.full(rays.shape[:2], np.inf)
    _face = np.zeros(rays.shape[:2], dtype=int)
    for _i, (_axis, _c, _) in enumerate(CORRIDOR):
        with np.errstate(divide="ignore"):
            _t = _c / rays[..., _axis]
        _hit = (_t > 0) & (_t < depth)
        depth[_hit] = _t[_hit]
        _face[_hit] = _i
    _pts = rays * depth[..., None]  # rays have x = 1, so the ray parameter is the depth

    # 25 cm checkerboard on each face, from the two coordinates along the face:
    _idx = np.floor(_pts / 0.25).astype(int)
    _normal = np.array([a for a, _, _ in CORRIDOR])[_face]
    _parity = (_idx.sum(axis=-1) - np.take_along_axis(_idx, _normal[..., None], axis=-1)[..., 0]) % 2
    rgb = (np.array([c for _, _, c in CORRIDOR])[_face] * (0.55 + 0.45 * _parity[..., None])).astype(np.uint8)
    return CORRIDOR, depth, rgb


@app.cell
def _(CImage, TColor, TColormap, TPixelCoord, camera, colormap, depth, fov_h, model, np, rgb):
    MAX_DEPTH = 6.0  # m, for the colormap

    image = CImage.from_numpy(rgb)
    _pp = TPixelCoord(round(camera.cx), round(camera.cy))
    image.drawCircle(_pp, 6, TColor(220, 0, 0), 2)  # principal point
    image.textOut(TPixelCoord(4, 4), f"{model.value}  HFOV {fov_h:.0f} deg", TColor(0, 0, 0))

    # Depth image, colored through a 256-entry lookup table built with an MRPT colormap:
    _lut = np.array([[c.R, c.G, c.B] for c in (colormap(TColormap.cmJET, i / 255) for i in range(256))])
    _levels = np.clip(255 * depth / MAX_DEPTH, 0, 255).astype(int)
    depth_image = CImage.from_numpy((255 * _lut[_levels]).astype(np.uint8))
    depth_image.textOut(TPixelCoord(4, 4), f"depth: 0 (blue) to {MAX_DEPTH} m (red)", TColor(255, 255, 255))
    return depth_image, image


@app.cell(hide_code=True)
def _(camera, depth_image, fov_h, image, mo, np, plt):
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for _ax, _img, _title in [(_axs[0], image, "Camera image (CImage)"), (_axs[1], depth_image, "Depth image")]:
        _ax.imshow(_img.as_numpy())  # zero-copy NumPy view of the CImage
        _ax.set_title(_title)
        _ax.set_xlabel("u (px)")
        _ax.set_ylabel("v (px)")
    _fig.tight_layout()

    _K = np.array(camera.intrinsicParams())
    mo.vstack(
        [
            _fig,
            mo.hstack(
                [
                    mo.md(
                        f"""
    **Intrinsic matrix** `TCamera.intrinsicParams()`:

    | | | |
    |---|---|---|
    | {_K[0, 0]:.1f} | {_K[0, 1]:.1f} | {_K[0, 2]:.1f} |
    | {_K[1, 0]:.1f} | {_K[1, 1]:.1f} | {_K[1, 2]:.1f} |
    | {_K[2, 0]:.1f} | {_K[2, 1]:.1f} | {_K[2, 2]:.1f} |

    Horizontal field of view (from the pixel rays): **{fov_h:.1f}°**
    """
                    ),
                    mo.md(f"**`print(camera)`**:\n\n```ini\n{camera}\n```"),
                ],
                justify="start",
                gap=3,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. From a depth image to a point cloud

    Depth cameras deliver an image like the one above, and `unprojectInto()`
    turns it into a point cloud using the camera intrinsics. Below, the same
    depth image is unprojected twice: with the right model, and with the
    distortion ignored (as with an uncalibrated camera). Each pixel keeps its
    depth, but its ray is wrong, so points move sideways: walls seen at an angle
    and the floor bend, and the corridor shrinks or grows. With the `none` model
    both clouds are identical.
    """)
    return


@app.cell
def _(camera, depth, np, pinhole, unproject):
    _d = depth.astype(np.float32)
    cloud = unproject(camera, _d)
    cloud_pinhole = unproject(pinhole, _d)
    cloud_error = np.linalg.norm(cloud_pinhole - cloud, axis=1)  # the same pixels, in the same order
    return cloud, cloud_error, cloud_pinhole


@app.cell(hide_code=True)
def _(CORRIDOR, cloud, cloud_error, cloud_pinhole, mo, np, plt):
    # Two thin slices of the cloud, selected with the right model (both clouds share pixels and depths):
    _slices = [
        ("Top view, horizontal slice (|z| < 5 cm)", (1, 0), np.abs(cloud[:, 2]) < 0.05),
        ("Front view, slice at x = 3 m (± 10 cm)", (1, 2), np.abs(cloud[:, 0] - 3.0) < 0.1),
    ]
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for _ax, (_title, (_i, _j), _in) in zip(_axs, _slices):
        _ax.scatter(cloud_pinhole[_in, _i], cloud_pinhole[_in, _j], s=3, color="#eb6834", label="Distortion ignored")
        _ax.scatter(cloud[_in, _i], cloud[_in, _j], s=3, color="#2a78d6", label="Right camera model")
        for _axis, _c, _ in CORRIDOR:  # corridor faces seen in this view, as gray lines
            if _axis == _i:
                _ax.axvline(_c, color="0.6", lw=1, ls="--", zorder=0)
            elif _axis == _j:
                _ax.axhline(_c, color="0.6", lw=1, ls="--", zorder=0)
        _ax.set_title(_title)
        _ax.set_aspect("equal", adjustable="datalim")
        _ax.invert_xaxis()  # y points left
        _ax.grid(alpha=0.3)
    _axs[0].set_xlabel("y, left (m)")
    _axs[0].set_ylabel("x, forward (m)")
    _axs[1].set_xlabel("y, left (m)")
    _axs[1].set_ylabel("z, up (m)")
    _axs[0].legend(loc="lower left", fontsize=8, markerscale=4)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"Ignoring the distortion moves the points by **{100 * cloud_error.mean():.1f} cm** "
                f"on average, and up to **{100 * cloud_error.max():.1f} cm** ({len(cloud):,} points)."
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/camera_models.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
