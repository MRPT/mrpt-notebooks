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
    # Odometry uncertainty: Gaussian vs. particles

    A robot drives with noisy wheel odometry, so the uncertainty in its pose
    grows at every step. This notebook uses the [MRPT](https://www.mrpt.org/)
    Python wheel (`pip install mrpt3`) to propagate that uncertainty in SE(2)
    in two ways:

    - **Linearized Gaussian**: Gaussian poses composed with first-order
      (Jacobian) error propagation, as an EKF does.
    - **Monte Carlo**: thousands of particles, each moved with its own random
      odometry error, as a particle filter does.

    It uses these MRPT modules:

    - `mrpt.obs`: the probabilistic odometry motion model (`CActionRobotMovement2D`).
    - `mrpt.poses`: poses and Gaussian pose PDFs (`CPose2D`, `CPose3D`, `CPose3DPDFGaussian`).
    - `mrpt.slam` and `mrpt.bayes`: a particle filter (`CMonteCarloLocalization2D`, `CParticleFilter`).

    **Move the sliders**: with little heading noise both methods agree, but
    with long paths and large heading noise the particles bend into a
    "banana" that no ellipse can describe.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mrpt.bayes import CParticleFilter
    from mrpt.math import CMatrixDouble66, TPose2D
    from mrpt.obs import CActionCollection, CActionRobotMovement2D
    from mrpt.poses import CPose2D, CPose3D, CPose3DPDFGaussian
    from mrpt.random import getRandomGenerator
    from mrpt.slam import CMonteCarloLocalization2D
    return (
        CActionCollection,
        CActionRobotMovement2D,
        CMatrixDouble66,
        CMonteCarloLocalization2D,
        CParticleFilter,
        CPose2D,
        CPose3D,
        CPose3DPDFGaussian,
        TPose2D,
        getRandomGenerator,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    n_steps = mo.ui.slider(start=1, stop=60, step=1, value=40, label="Number of steps", show_value=True)
    turn_rate = mo.ui.slider(start=-15, stop=15, step=0.5, value=1, label="Turn rate (deg/step)", show_value=True)
    trans_noise = mo.ui.slider(
        start=0.5, stop=20, step=0.5, value=2, label="Translation noise σ (% of distance)", show_value=True
    )
    rot_noise = mo.ui.slider(
        start=0.5, stop=20, step=0.5, value=10, label="Heading noise σ (deg per meter)", show_value=True
    )

    mo.vstack(
        [
            mo.md("### Parameters (each step: 0.5 m forward, then turn)"),
            mo.hstack([n_steps, turn_rate], justify="start", gap=2),
            mo.hstack([trans_noise, rot_noise], justify="start", gap=2),
        ]
    )
    return n_steps, rot_noise, trans_noise, turn_rate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The motion model

    At each step the odometry reads the same increment: 0.5 m forward plus the
    turn. MRPT's Gaussian motion model turns that reading into a PDF over the
    true increment, whose standard deviations grow with the distance $d$
    traveled: $\sigma_{xy} = a_1 d$ and $\sigma_\phi = a_3 d$.
    """)
    return


@app.cell
def _(CActionCollection, CActionRobotMovement2D, CPose2D, mo, np, rot_noise, trans_noise, turn_rate):
    STEP_LENGTH = 0.5  # meters
    odometry = CPose2D(STEP_LENGTH, 0.0, np.radians(turn_rate.value))

    MotionOptions = CActionRobotMovement2D.TMotionModelOptions
    motion_opts = MotionOptions()
    motion_opts.modelSelection = CActionRobotMovement2D.mmGaussian
    # Arguments: a1, a2, a3, a4, minStdXY, minStdPHI (only a1 and a3 are used here)
    motion_opts.gaussianModel = MotionOptions.TOptions_GaussianModel(
        trans_noise.value / 100.0, 0.0, np.radians(rot_noise.value), 0.0, 0.0, 0.0
    )

    action = CActionRobotMovement2D()
    action.computeFromOdometry(odometry, motion_opts)
    step_pdf = action.poseChange  # a CPosePDFGaussian over the true increment

    # The particle filter takes a collection of actions (here, just one):
    actions = CActionCollection()
    actions.insert(action)
    mo.md(f"Per-step increment, as a `{type(step_pdf).__name__}`:\n```\n{step_pdf}\n```")
    return actions, step_pdf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Linearized Gaussian propagation

    If the pose at step $k-1$ is Gaussian, $\mathcal{N}(\mu_{k-1}, \Sigma_{k-1})$,
    composing it with the increment $u \sim \mathcal{N}(\bar u, \Sigma_u)$ gives

    $$\mu_k = \mu_{k-1} \oplus \bar u, \qquad \Sigma_k = J_1 \Sigma_{k-1} J_1^\top + J_2 \Sigma_u J_2^\top,$$

    with $J_1, J_2$ the Jacobians of the composition $\oplus$. The `+` operator
    of `CPose3DPDFGaussian` implements this. A planar pose is just a 3D pose
    with $z$, pitch and roll equal to zero, so we lift the 2D increment to 3D.
    """)
    return


@app.cell
def _(CMatrixDouble66, CPose3D, CPose3DPDFGaussian, n_steps, np, step_pdf):
    XY_YAW = [0, 1, 3]  # indices of (x, y, yaw) in the 6D vector (x, y, z, yaw, pitch, roll)

    def to_3d(pdf2d):
        """A 2D Gaussian pose PDF as a CPose3DPDFGaussian with z = pitch = roll = 0."""
        cov6 = np.zeros((6, 6))
        cov6[np.ix_(XY_YAW, XY_YAW)] = np.asarray(pdf2d.getCovariance())
        return CPose3DPDFGaussian(CPose3D(pdf2d.getMean()), CMatrixDouble66(cov6))

    step_pdf_3d = to_3d(step_pdf)
    gauss = CPose3DPDFGaussian(CPose3D())  # start at the origin, without uncertainty
    gauss_means = [np.zeros(3)]
    gauss_covs = [np.zeros((3, 3))]
    for _ in range(n_steps.value):
        gauss = gauss + step_pdf_3d  # linearized pose composition
        gauss_means.append(np.array([gauss.mean.x, gauss.mean.y, gauss.mean.yaw]))
        gauss_covs.append(np.asarray(gauss.cov)[np.ix_(XY_YAW, XY_YAW)])
    return gauss_covs, gauss_means


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Monte Carlo propagation

    A particle filter without observations only runs its prediction step: each
    particle is composed with its own random sample of the increment, so no
    linearization is involved. `CParticleFilter.executeOn()` does that on a
    `CMonteCarloLocalization2D` particle set when no observation is given.
    """)
    return


@app.cell
def _(
    CMonteCarloLocalization2D,
    CParticleFilter,
    TPose2D,
    actions,
    getRandomGenerator,
    n_steps,
    np,
):
    N_PARTICLES = 2000
    getRandomGenerator().randomize(42)  # MRPT's global random generator, for repeatable results

    particles = CMonteCarloLocalization2D(N_PARTICLES)
    particles.resetDeterministic(TPose2D(0.0, 0.0, 0.0), N_PARTICLES)
    pf = CParticleFilter()  # defaults: standard proposal, fixed number of particles

    snapshots = []  # particle (x, y, phi) after each step
    for _ in range(n_steps.value):
        pf.executeOn(particles, actions, None)  # prediction only: no observation
        snapshots.append(particles.getParticlesAsNumpy()[:, :3])

    _cov, _mean = particles.getCovarianceAndMean()
    pf_mean = np.array([_mean.x, _mean.y, _mean.phi])
    pf_cov = np.asarray(_cov)
    return pf_cov, pf_mean, snapshots


@app.cell(hide_code=True)
def _(gauss_covs, gauss_means, mo, n_steps, np, pf_cov, pf_mean, snapshots):
    # Fraction of particles inside the (x, y) 95% ellipse of the Gaussian:
    _d = snapshots[-1][:, :2] - gauss_means[-1][:2]
    _m2 = np.einsum("ij,jk,ik->i", _d, np.linalg.inv(gauss_covs[-1][:2, :2]), _d)
    _inside = 100.0 * np.mean(_m2 < 5.991)  # chi-square, 2 DOF, 95%
    _sg = np.sqrt(np.diag(gauss_covs[-1]))
    _sp = np.sqrt(np.diag(pf_cov))
    _kind = "success" if _inside > 92.0 else ("warn" if _inside > 85.0 else "danger")
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"**{_inside:.1f}%** of the particles fall inside the 95% ellipse of the "
                    "linearized Gaussian (it would be 95% if the true PDF were Gaussian)."
                ),
                kind=_kind,
            ),
            mo.md(
                f"""
    Pose after {n_steps.value} steps:

    |  | x (m) | y (m) | yaw (deg) | σx (m) | σy (m) | σyaw (deg) |
    |---|---|---|---|---|---|---|
    | Linearized Gaussian | {gauss_means[-1][0]:.3f} | {gauss_means[-1][1]:.3f} | {np.degrees(gauss_means[-1][2]):.1f} | {_sg[0]:.3f} | {_sg[1]:.3f} | {np.degrees(_sg[2]):.2f} |
    | Particles (sample mean and std) | {pf_mean[0]:.3f} | {pf_mean[1]:.3f} | {np.degrees(pf_mean[2]):.1f} | {_sp[0]:.3f} | {_sp[1]:.3f} | {np.degrees(_sp[2]):.2f} |
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(gauss_covs, gauss_means, np, pf_cov, pf_mean, plt, snapshots):
    def ellipse(mean, cov, chi2=5.991):
        """Points on the (x, y) 95% confidence ellipse of a Gaussian."""
        vals, vecs = np.linalg.eigh(cov[:2, :2])
        t = np.linspace(0.0, 2.0 * np.pi, 100)
        pts = vecs @ (np.sqrt(np.maximum(vals, 0.0) * chi2)[:, None] * np.stack([np.cos(t), np.sin(t)]))
        return pts[0] + mean[0], pts[1] + mean[1]

    path = np.array(gauss_means)
    n = len(snapshots)
    shown = sorted({*range(n - 1, -1, -max(1, n // 5))})  # about 5 particle snapshots

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axs[0]
    ax.plot(*path[:, :2].T, "k.-", lw=1, ms=3, label="Noise-free path (Gaussian mean)")
    for k in shown:
        ax.scatter(*snapshots[k][:, :2].T, s=1, color="tab:red", alpha=0.3, lw=0)
        ax.plot(*ellipse(gauss_means[k + 1], gauss_covs[k + 1]), color="tab:blue", lw=1.2)
    ax.set_title("Uncertainty along the path")

    ax = axs[1]
    last = snapshots[-1]
    ax.scatter(*last[:, :2].T, s=3, color="tab:red", alpha=0.35, lw=0)
    ax.plot(*ellipse(gauss_means[-1], gauss_covs[-1]), color="tab:blue", lw=2, label="Linearized Gaussian, 95%")
    ax.plot(*ellipse(pf_mean, pf_cov), color="tab:red", lw=2, ls="--", label="Particle covariance, 95%")
    ax.plot(*gauss_means[-1][:2], "x", color="tab:blue", ms=10, mew=2, label="Gaussian mean")
    ax.plot(*pf_mean[:2], "+", color="tab:red", ms=12, mew=2, label="Particle mean")
    ax.scatter([], [], s=12, color="tab:red", label="Particles")
    ax.set_title(f"Final pose (step {n})")

    for ax in axs:
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.3)
    handles, labels = axs[0].get_legend_handles_labels()
    handles2, labels2 = axs[1].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look at**

    - The Gaussian mean stays on the noise-free path, while the particle mean
      is pulled to the inner side of the banana: the mean of a curved
      distribution is not on the curve.
    - The ellipse has roughly the right spread, but it cannot bend: it covers
      poses that no particle reached, and misses particles on the tips.
    - Heading noise is what bends the cloud, and the error it causes grows
      with the distance traveled. Set a small heading noise and the two
      methods agree.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/pose_uncertainty.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
