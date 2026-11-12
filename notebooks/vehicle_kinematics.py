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
    # Mobile robot kinematics: differential drive vs. holonomic

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to simulate how two kinds of mobile robots move when
    they receive velocity commands, and how uncertain dead reckoning
    (odometry) becomes over time. It uses three MRPT modules:

    - `mrpt.kinematics`: velocity commands (`CVehicleVelCmd_DiffDriven`,
      `CVehicleVelCmd_Holo`) and kinematic simulators
      (`CVehicleSimul_DiffDriven`, `CVehicleSimul_Holo`).
    - `mrpt.obs`: the probabilistic odometry motion model used by particle
      filters (`CActionRobotMovement2D`).
    - `mrpt.poses` and `mrpt.random`: SE(2) poses (`CPose2D`) and MRPT's
      random generator, seeded for repeatable results.

    **Move the sliders**: every plot is recomputed. Try a square path with
    both robots to see how differently they move.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mrpt.kinematics import (
        CVehicleSimul_DiffDriven,
        CVehicleSimul_Holo,
        CVehicleVelCmd_DiffDriven,
        CVehicleVelCmd_Holo,
    )
    from mrpt.obs import CActionRobotMovement2D
    from mrpt.poses import CPose2D
    from mrpt.random import getRandomGenerator
    return (
        CActionRobotMovement2D,
        CPose2D,
        CVehicleSimul_DiffDriven,
        CVehicleSimul_Holo,
        CVehicleVelCmd_DiffDriven,
        CVehicleVelCmd_Holo,
        getRandomGenerator,
        mo,
        np,
        plt,
    )


@app.cell
def _(np):
    def velocities(log):
        """Linear speed in the robot frame (vx, vy) and angular speed, by finite
        differences of a log of poses with rows (t, x, y, phi)."""
        t, x, y, phi = log.T
        dt = np.diff(t)
        dx, dy = np.diff(x) / dt, np.diff(y) / dt
        c, s = np.cos(phi[:-1]), np.sin(phi[:-1])
        w = np.angle(np.exp(1j * np.diff(phi))) / dt  # wrapped to [-pi, pi]
        return t[1:], c * dx + s * dy, -s * dx + c * dy, w

    def draw_headings(ax, log, every, color):
        """Arrows showing the robot heading every few samples."""
        p = log[::every]
        ax.quiver(p[:, 1], p[:, 2], np.cos(p[:, 3]), np.sin(p[:, 3]), color=color, width=0.004, scale=25, zorder=3)
    return draw_headings, velocities


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Differential-drive robot

    A robot with two independently driven wheels (like most indoor robots and
    robot vacuums) is commanded with a **linear speed** $v$ and an
    **angular speed** $\omega$. It cannot move sideways: keeping $(v, \omega)$
    constant drives it along a circular arc of radius $R = v / \omega$, and
    $v = 0$ makes it spin in place. Any path is a sequence of such arcs.

    Below, a short program of `CVehicleVelCmd_DiffDriven` commands is sent to
    the simulator `CVehicleSimul_DiffDriven` with `movementCommand(v, ω)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    maneuver = mo.ui.dropdown(options=["Arc", "Slalom", "Square"], value="Slalom", label="Maneuver")
    lin_vel = mo.ui.slider(start=0.1, stop=1.5, step=0.1, value=0.5, label="Linear speed v (m/s)", show_value=True)
    ang_vel = mo.ui.slider(start=10, stop=90, step=5, value=30, label="Angular speed ω (deg/s)", show_value=True)
    seg_time = mo.ui.slider(start=1.0, stop=10.0, step=0.5, value=4.0, label="Duration of each command (s)", show_value=True)

    mo.vstack(
        [
            mo.md("### Velocity commands"),
            mo.hstack([maneuver, seg_time], justify="start", gap=2),
            mo.hstack([lin_vel, ang_vel], justify="start", gap=2),
        ]
    )
    return ang_vel, lin_vel, maneuver, seg_time


@app.cell
def _(CVehicleVelCmd_DiffDriven, ang_vel, lin_vel, maneuver, np, seg_time):
    def diff_cmd(v, w):
        cmd = CVehicleVelCmd_DiffDriven()
        cmd.lin_vel = v  # m/s
        cmd.ang_vel = w  # rad/s
        return cmd

    _v, _w, _T = lin_vel.value, np.radians(ang_vel.value), seg_time.value

    # The program: a list of (command, duration in seconds).
    if maneuver.value == "Arc":
        diff_program = [(diff_cmd(_v, _w), _T)]
    elif maneuver.value == "Slalom":
        # Alternate left and right turns. Half-length first and last turns end with the initial heading.
        diff_program = [(diff_cmd(_v, _w * (-1) ** i), _T * (0.5 if i in (0, 5) else 1.0)) for i in range(6)]
    else:
        # Four straight sides, each followed by a 90 deg turn in place.
        diff_program = [(diff_cmd(_v, 0.0), _T), (diff_cmd(0.0, _w), (np.pi / 2) / _w)] * 4
    return (diff_program,)


@app.cell
def _(CVehicleSimul_DiffDriven, diff_program, np):
    _sim = CVehicleSimul_DiffDriven()
    _dt = 0.05  # logging period (s); the simulator integrates internally at 2 kHz
    _t = 0.0
    _log = [(0.0, 0.0, 0.0, 0.0)]
    for _cmd, _duration in diff_program:
        _sim.movementCommand(_cmd.lin_vel, _cmd.ang_vel)
        _n = int(np.ceil(_duration / _dt - 1e-9))
        for _i in range(_n):
            _step = min(_dt, _duration - _i * _dt)  # shorter last step, to end exactly on time
            _sim.simulateOneTimeStep(_step)
            _t += _step
            _p = _sim.getCurrentOdometricPose()
            _log.append((_t, _p.x, _p.y, _p.phi))
    diff_log = np.array(_log)  # rows: (t, x, y, phi)
    return (diff_log,)


@app.cell(hide_code=True)
def _(diff_log, diff_program, draw_headings, np, plt, velocities):
    _fig, _axs = plt.subplot_mosaic([["xy", "v"], ["xy", "w"]], figsize=(12, 5.5), width_ratios=[1.3, 1])
    _ax = _axs["xy"]
    _ax.plot(diff_log[:, 1], diff_log[:, 2], color="tab:blue", lw=2, label="Path")
    draw_headings(_ax, diff_log, max(1, len(diff_log) // 25), "tab:blue")
    _ax.plot(0, 0, "ko", ms=6, label="Start")
    _ax.plot(diff_log[-1, 1], diff_log[-1, 2], "s", color="tab:red", ms=7, label="End")
    _ax.set(xlabel="x (m)", ylabel="y (m)", title="Path and heading (arrows)")
    _ax.set_aspect("equal", adjustable="datalim")
    _ax.margins(0.1)
    _ax.grid(alpha=0.3)
    _ax.legend(loc="best", fontsize=8)

    # Velocities measured from the simulated motion, with the commands as dashed lines:
    _t, _vx, _vy, _w = velocities(diff_log)
    _tc = np.cumsum([0.0] + [d for _, d in diff_program])
    _vc = [c.lin_vel for c, _ in diff_program]
    _wc = [np.degrees(c.ang_vel) for c, _ in diff_program]
    _axs["v"].plot(_t, _vx, color="tab:blue", label="Simulated")
    _axs["v"].stairs(_vc, _tc, baseline=None, color="black", ls="--", lw=1, label="Commanded")
    _axs["v"].set(ylabel="v (m/s)", title="Velocities", ylim=(-0.1, 1.2 * max(_vc)))
    _axs["v"].legend(loc="lower right", fontsize=8)
    _axs["w"].plot(_t, np.degrees(_w), color="tab:orange")
    _axs["w"].stairs(_wc, _tc, baseline=None, color="black", ls="--", lw=1)
    _wmax = 1.2 * max(np.abs(_wc))
    _axs["w"].set(xlabel="Time (s)", ylabel="ω (deg/s)", ylim=(-_wmax, _wmax))
    _axs["w"].sharex(_axs["v"])
    for _k in ("v", "w"):
        _axs[_k].grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Odometry uncertainty

    The robot above measures its motion by counting wheel turns (odometry),
    but real wheels slip and their radii are never exactly known, so small
    errors accumulate. The simulator integrates the commands perfectly (its
    own odometry error model is not exposed to Python), so here we take its
    output as the odometry readings and ask: *where may the robot really be?*

    Particle filters answer that with a probabilistic **motion model**. For each
    odometry increment (every 0.5 s), `CActionRobotMovement2D.computeFromOdometry()`
    sets up the model of Thrun et al. ("Probabilistic Robotics"), and
    `drawSingleSample()` draws a random version of the true motion. Chaining
    those samples gives possible true paths. Note how small heading errors
    grow into large lateral errors as the robot keeps moving.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rot_err = mo.ui.slider(start=0.0, stop=30.0, step=1.0, value=5.0, label="Rotation error per rotation α1 (%)", show_value=True)
    rot_per_m = mo.ui.slider(start=0.0, stop=10.0, step=0.5, value=2.0, label="Rotation error per meter α2 (deg/m)", show_value=True)
    trans_err = mo.ui.slider(start=0.0, stop=20.0, step=1.0, value=2.0, label="Translation error per meter α3 (%)", show_value=True)
    n_samples = mo.ui.slider(start=10, stop=300, step=10, value=100, label="Number of samples", show_value=True)

    mo.vstack(
        [
            mo.md("### Motion model (1σ errors)"),
            mo.hstack([rot_err, rot_per_m], justify="start", gap=2),
            mo.hstack([trans_err, n_samples], justify="start", gap=2),
        ]
    )
    return n_samples, rot_err, rot_per_m, trans_err


@app.cell
def _(
    CActionRobotMovement2D,
    CPose2D,
    diff_log,
    getRandomGenerator,
    n_samples,
    np,
    rot_err,
    rot_per_m,
    trans_err,
):
    _opts = CActionRobotMovement2D.TMotionModelOptions()
    _opts.modelSelection = CActionRobotMovement2D.mmThrun
    _opts.thrunModel.alfa1_rot_rot = rot_err.value / 100
    _opts.thrunModel.alfa2_rot_trans = np.radians(rot_per_m.value)
    _opts.thrunModel.alfa3_trans_trans = trans_err.value / 100

    getRandomGenerator().randomize(42)  # repeatable samples
    _every = np.unique(np.r_[0 : len(diff_log) : 10, len(diff_log) - 1])  # every 0.5 s, and the last one
    _odometry = [CPose2D(x, y, phi) for _, x, y, phi in diff_log[_every]]
    _action = CActionRobotMovement2D()
    _samples = [CPose2D() for _ in range(n_samples.value)]
    true_paths = np.zeros((len(_odometry), n_samples.value, 2))  # possible (x, y) of the robot
    for _i in range(1, len(_odometry)):
        _action.computeFromOdometry(_odometry[_i] - _odometry[_i - 1], _opts)  # pose increment
        for _j in range(n_samples.value):
            _samples[_j] = _samples[_j] + _action.drawSingleSample()  # SE(2) composition
            true_paths[_i, _j] = _samples[_j].x, _samples[_j].y
    final_headings = np.array([s.phi for s in _samples])
    return final_headings, true_paths


@app.cell(hide_code=True)
def _(diff_log, final_headings, mo, np, true_paths):
    _end = true_paths[-1]
    _dist = np.sum(np.hypot(np.diff(diff_log[:, 1]), np.diff(diff_log[:, 2])))
    _err = np.hypot(_end[:, 0] - diff_log[-1, 1], _end[:, 1] - diff_log[-1, 2])
    _head = np.degrees(np.angle(np.exp(1j * (final_headings - diff_log[-1, 3]))))
    mo.md(
        f"After **{diff_log[-1, 0]:.0f} s** and **{_dist:.1f} m**, the true position differs from "
        f"odometry by **{np.median(_err):.2f} m** (median; worst sample: {_err.max():.2f} m), "
        f"and the heading by **{np.std(_head):.1f}°** (1σ)."
    )
    return


@app.cell(hide_code=True)
def _(diff_log, plt, true_paths):
    _fig, _ax = plt.subplots(figsize=(9, 6))
    _ax.plot(true_paths[:, :, 0], true_paths[:, :, 1], color="tab:orange", lw=0.6, alpha=0.25)
    _ax.scatter(true_paths[-1, :, 0], true_paths[-1, :, 1], s=10, color="tab:red", zorder=3, label="Possible final positions")
    _ax.plot(diff_log[:, 1], diff_log[:, 2], color="tab:blue", lw=2.5, label="Odometry")
    _ax.plot(diff_log[-1, 1], diff_log[-1, 2], "s", color="tab:blue", ms=8, label="Odometry final position")
    _ax.plot([], [], color="tab:orange", label="Possible true paths")
    _ax.set(xlabel="x (m)", ylabel="y (m)", title="Odometry vs. possible ground-truth paths")
    _ax.set_aspect("equal", adjustable="datalim")
    _ax.grid(alpha=0.3)
    _ax.legend(loc="best", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Holonomic robot

    An omnidirectional robot (e.g. with mecanum or omni wheels) can move in
    **any direction regardless of its heading**. `CVehicleSimul_Holo` models
    one whose commands (`CVehicleVelCmd_Holo`) are a speed, a direction of
    motion (relative to the robot heading), a *ramp time* during which the
    velocity changes smoothly from the current to the new one, and a rotation
    speed used to turn the robot towards the motion direction.

    Here a simple controller drives the robot through the corners of a 4 m
    square: every 0.1 s it sends a command towards the next corner. A long ramp
    time rounds the corners (and may overshoot them), while the rotation speed
    sets how fast the heading follows the motion. With zero rotation speed,
    the robot slides around the square always facing the same way.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    holo_speed = mo.ui.slider(start=0.2, stop=2.0, step=0.1, value=1.0, label="Speed (m/s)", show_value=True)
    ramp_time = mo.ui.slider(start=0.1, stop=3.0, step=0.1, value=0.5, label="Ramp time (s)", show_value=True)
    rot_speed = mo.ui.slider(start=0, stop=180, step=10, value=60, label="Rotation speed (deg/s)", show_value=True)

    mo.vstack([mo.md("### Holonomic commands"), mo.hstack([holo_speed, ramp_time, rot_speed], justify="start", gap=2)])
    return holo_speed, ramp_time, rot_speed


@app.cell
def _(CVehicleSimul_Holo, CVehicleVelCmd_Holo, holo_speed, np, ramp_time, rot_speed):
    corners = np.array([[4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]])
    _sim = CVehicleSimul_Holo()
    _dt = 0.1  # control period (s)
    _target = 0
    _log = [(0.0, 0.0, 0.0, 0.0)]
    for _i in range(1, 1200):  # 120 s at most
        _p = _sim.getCurrentOdometricPose()
        _dx, _dy = corners[_target] - (_p.x, _p.y)
        if np.hypot(_dx, _dy) < 0.25:  # corner reached
            _target += 1
            if _target == len(corners):
                break
            _dx, _dy = corners[_target] - (_p.x, _p.y)
        # The command uses the direction relative to the robot heading:
        _cmd = CVehicleVelCmd_Holo(holo_speed.value, np.arctan2(_dy, _dx) - _p.phi, ramp_time.value, np.radians(rot_speed.value))
        # The simulator takes it in the odometry frame (as the C++ sendVelCmd() does):
        _sim.sendVelRampCmd(_cmd.vel, _cmd.dir_local + _p.phi, _cmd.ramp_time, _cmd.rot_speed)
        _sim.simulateOneTimeStep(_dt)
        _q = _sim.getCurrentGTPose()
        _log.append((_i * _dt, _q.x, _q.y, _q.phi))
    holo_log = np.array(_log)  # rows: (t, x, y, phi)
    holo_done = _target == len(corners)
    return corners, holo_done, holo_log


@app.cell(hide_code=True)
def _(holo_done, holo_log, mo):
    mo.md(
        f"The holonomic robot went around the square in **{holo_log[-1, 0]:.1f} s**."
        if holo_done
        else f"The robot did not finish the square within {holo_log[-1, 0]:.0f} s."
    )
    return


@app.cell(hide_code=True)
def _(corners, draw_headings, holo_log, np, plt, velocities):
    _fig, _axs = plt.subplot_mosaic([["xy", "v"], ["xy", "w"]], figsize=(12, 5.5), width_ratios=[1.3, 1])
    _ax = _axs["xy"]
    _ax.plot(*np.vstack([[0, 0], corners]).T, color="0.7", ls="--", lw=1, label="Square")
    _ax.plot(holo_log[:, 1], holo_log[:, 2], color="tab:green", lw=2, label="Path")
    draw_headings(_ax, holo_log, 10, "tab:green")
    _ax.plot(*corners.T, "ko", ms=5, label="Corners")
    _ax.set(xlabel="x (m)", ylabel="y (m)", title="Path and heading (arrows, every 1 s)")
    _ax.set_aspect("equal", adjustable="datalim")
    _ax.margins(0.1)
    _ax.grid(alpha=0.3)
    _ax.legend(loc="center", fontsize=8)

    _t, _vx, _vy, _w = velocities(holo_log)
    _axs["v"].plot(_t, _vx, color="tab:green", label="Forward ($v_x$)")
    _axs["v"].plot(_t, _vy, color="tab:purple", label="Sideways ($v_y$)")
    _axs["v"].set(ylabel="Speed (m/s)", title="Velocities in the robot frame")
    _axs["v"].legend(loc="best", fontsize=8)
    _axs["w"].plot(_t, np.degrees(_w), color="tab:orange")
    _axs["w"].set(xlabel="Time (s)", ylabel="ω (deg/s)")
    _axs["w"].sharex(_axs["v"])
    for _k in ("v", "w"):
        _axs[_k].grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/vehicle_kinematics.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
