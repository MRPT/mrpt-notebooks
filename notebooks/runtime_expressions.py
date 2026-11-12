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
    # Math formulas compiled at runtime

    This notebook uses the [MRPT](https://www.mrpt.org/) Python wheel
    (`pip install mrpt3`) to compile and evaluate math expressions given as
    text, at runtime. MRPT wraps the [exprtk](https://github.com/ArashPartow/exprtk)
    library, which turns a string (from a simple formula to a small program with
    variables, `if`/`else` and loops) into an expression tree that is evaluated in
    C++. MRPT uses it, for example, to let users write formulas in configuration
    files. It uses two MRPT modules:

    - `mrpt.expr`: the expression compiler `CRuntimeCompiledExpression`.
    - `mrpt.config`: configuration files in memory (`CConfigFileMemory`), with
      `@define` variables and `$eval{...}` formulas.

    **Edit the formulas and move the sliders**: everything below is recomputed.
    """)
    return


@app.cell
def _():
    import re
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mrpt.config import CConfigFileMemory
    from mrpt.expr import CRuntimeCompiledExpression
    return CConfigFileMemory, CRuntimeCompiledExpression, mo, np, plt, re, time


@app.cell
def _(CRuntimeCompiledExpression, re):
    def compile_formula(formula, variables):
        """Compiles a formula, given the values of its variables as a dict."""
        expr = CRuntimeCompiledExpression()
        expr.compile(formula, variables)
        return expr

    def error_message(exc):
        """The relevant part of an MRPT exception message (without backticks, to show it as code)."""
        msg = str(exc).splitlines()[0]
        for pattern in (r"Error: `(.*)`", r"Message:\s*(.*)"):
            if match := re.search(pattern, str(exc)):
                msg = match.group(1)
                break
        return msg.replace("`", "'")
    return compile_formula, error_message


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Plot a formula

    Write a formula of $x$ and three parameters $a$, $b$, $c$, controlled by
    the sliders. It may use the usual math functions (`sin`, `exp`, `sqrt`,
    `abs`, `clamp`, `hypot`, ...), the constant `pi`, conditionals, and even
    loops (statements are separated with `;`, also after a closing `}`).
    Pick an example to start with.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    examples = {
        "Damped oscillation": ("a * exp(-b * x) * sin(2 * pi * c * x)", (1.0, 0.3, 0.5), (0.0, 10.0)),
        "Gaussian bell": ("a * exp(-0.5 * ((x - b) / c)^2)", (1.0, 1.0, 1.5), (-5.0, 5.0)),
        "Huber robust loss (if / else)": (
            "if (abs(x) <= b) 0.5 * x^2;\nelse b * (abs(x) - 0.5 * b);",
            (0.0, 1.5, 0.0),
            (-5.0, 5.0),
        ),
        "Square wave as a Fourier series (loop)": (
            "var s := 0;\nfor (var k := 1; k <= 2 * a; k += 2) {\n  s += sin(k * x) / k;\n};\n4 / pi * s",
            (5.0, 0.0, 0.0),
            (-6.0, 6.0),
        ),
    }
    example = mo.ui.dropdown(options=list(examples), value="Damped oscillation", label="Example")
    example
    return example, examples


@app.cell(hide_code=True)
def _(example, examples, mo):
    _formula, (_a, _b, _c), _x_range = examples[example.value]
    formula = mo.ui.text_area(value=_formula, rows=5, label="Formula", full_width=True)
    a = mo.ui.slider(start=-10.0, stop=10.0, step=0.1, value=_a, label="a", show_value=True)
    b = mo.ui.slider(start=-10.0, stop=10.0, step=0.1, value=_b, label="b", show_value=True)
    c = mo.ui.slider(start=-10.0, stop=10.0, step=0.1, value=_c, label="c", show_value=True)
    x_range = mo.ui.range_slider(start=-10.0, stop=10.0, step=0.5, value=_x_range, label="x range", show_value=True)

    mo.vstack([formula, mo.hstack([a, b, c], justify="start", gap=2), x_range])
    return a, b, c, formula, x_range


@app.cell
def _(a, b, c, compile_formula, error_message, formula, np, time, x_range):
    xs = np.linspace(*x_range.value, 400)
    _t0 = time.perf_counter()
    try:
        # From Python, variable values are given when compiling, so we compile once per sample,
        # each time with a new object (re-compiling an existing one keeps its first variables).
        ys = np.array(
            [compile_formula(formula.value, {"x": x, "a": a.value, "b": b.value, "c": c.value}).eval() for x in xs]
        )
        formula_error = None
    except RuntimeError as _e:
        ys, formula_error = None, error_message(_e)
    eval_time = time.perf_counter() - _t0
    return eval_time, formula_error, xs, ys


@app.cell(hide_code=True)
def _(eval_time, formula, formula_error, mo, np, plt, xs, ys):
    if formula_error:
        _out = mo.callout(mo.md(f"**The formula does not compile**: `{formula_error}`"), kind="danger")
    else:
        _fig, _ax = plt.subplots(figsize=(10, 4))
        _ax.plot(xs, np.where(np.isfinite(ys), ys, np.nan), color="tab:blue", lw=2)
        _ax.set(xlabel="x", ylabel="f(x)", title=formula.value.replace("\n", " ")[:150])
        _ax.title.set_fontsize(9)
        _ax.grid(alpha=0.3)
        _fig.tight_layout()
        _out = mo.vstack(
            [
                _fig,
                mo.md(
                    f"{len(xs)} compilations and evaluations took **{1e3 * eval_time:.0f} ms** "
                    f"({1e6 * eval_time / len(xs):.0f} µs each)."
                ),
            ]
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Small programs, and speed

    The compiled expression is evaluated in C++, so a loop written *inside*
    the expression runs much faster than the same loop in pure Python. As a
    test, we approximate $\pi = \int_0^1 \frac{4}{1+x^2}\,dx$ with the midpoint
    rule and $n$ intervals, written as an exprtk program, as a Python loop,
    and with vectorized NumPy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_intervals = mo.ui.slider(
        steps=[10_000, 30_000, 100_000, 300_000, 1_000_000], value=100_000, label="Number of intervals n", show_value=True
    )
    n_intervals
    return (n_intervals,)


@app.cell
def _(compile_formula, n_intervals, np, time):
    pi_program = (
        "var s := 0;\n"
        "for (var i := 0; i < n; i += 1) {\n"
        "  var x := (i + 0.5) / n;\n"
        "  s += 4 / (1 + x^2);\n"
        "};\n"
        "s / n"
    )

    def pi_python(n):
        s = 0.0
        for i in range(n):
            x = (i + 0.5) / n
            s += 4 / (1 + x**2)
        return s / n

    def pi_numpy(n):
        x = (np.arange(n) + 0.5) / n
        return np.mean(4 / (1 + x**2))

    def timed(f, repeat=3):
        """Best run time of f() (in seconds), and its result."""
        best = np.inf
        for _ in range(repeat):
            t0 = time.perf_counter()
            value = f()
            best = min(best, time.perf_counter() - t0)
        return best, value

    _n = n_intervals.value
    _expr = compile_formula(pi_program, {"n": _n})
    timings = {
        "exprtk: compile": timed(lambda: compile_formula(pi_program, {"n": _n})),
        "exprtk: evaluate": timed(_expr.eval),
        "Pure Python": timed(lambda: pi_python(_n)),
        "NumPy": timed(lambda: pi_numpy(_n)),
    }
    return pi_program, timings


@app.cell(hide_code=True)
def _(mo, np, pi_program, plt, timings):
    _names = list(timings)
    _ms = np.array([1e3 * timings[k][0] for k in _names])
    _fig, _ax = plt.subplots(figsize=(7, 2.6))
    _ax.barh(_names, _ms, color=["tab:blue", "tab:blue", "tab:orange", "tab:green"], height=0.6)
    for _i, _v in enumerate(_ms):
        _ax.text(_v + 0.01 * _ms.max(), _i, f"{_v:.3g} ms", va="center", fontsize=9)
    _ax.set_xlim(0, _ms.max() * 1.25)
    _ax.invert_yaxis()
    _ax.set(xlabel="Time (ms)", title="Best of 3 runs")
    _ax.grid(alpha=0.3, axis="x")
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {k} | {1e3 * t:.3g} | {'' if v is None or k.endswith('compile') else f'{v:.10f}'} |"
        for k, (t, v) in timings.items()
    )
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(f"```\n{pi_program}\n```"),
                    mo.md(f"| Method | Time (ms) | Result |\n|---|---|---|\n{_rows}"),
                ]
            ),
            _fig,
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Formulas in configuration files

    MRPT configuration files (INI format, used by many MRPT applications)
    are preprocessed when loaded: `@define NAME value` defines a variable,
    `${NAME}` inserts its value, and `$eval{...}` evaluates a formula with this
    same expression engine. This keeps derived parameters consistent: change
    the wheel radius below and every speed limit follows. `CConfigFileMemory`
    parses the text, and `read_double()` returns the final values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    config_text = mo.ui.text_area(
        value="""# Parameters of a differential-drive robot
@define WHEEL_RADIUS 0.1
@define TRACK 0.45
@define MOTOR_RPM 3000
@define GEAR_RATIO 30
@define V_MAX $eval{2 * pi * WHEEL_RADIUS * MOTOR_RPM / GEAR_RATIO / 60}

[robot]
wheel_radius = ${WHEEL_RADIUS}
v_max = ${V_MAX}
w_max = $eval{2 * V_MAX / TRACK}
w_max_deg = $eval{rad2deg(2 * V_MAX / TRACK)}
time_for_10m = $eval{10 / V_MAX}
""",
        rows=14,
        label="Configuration file",
        full_width=True,
    )
    config_text
    return (config_text,)


@app.cell(hide_code=True)
def _(CConfigFileMemory, config_text, error_message, mo, re):
    try:
        _cfg = CConfigFileMemory(config_text.value)
        _raw = dict(re.findall(r"^\s*(\w+)\s*=\s*(.*?)\s*$", config_text.value, flags=re.MULTILINE))
        _rows = [
            f"| `[{s}]` | `{k}` | `{_raw.get(k, '')}` | `{_cfg.read_string(s, k)}` | {_cfg.read_double(s, k, 0.0):.6g} |"
            for s in _cfg.getAllSections()
            for k in _cfg.getAllKeys(s)
        ]
        _out = mo.md(
            "| Section | Key | In the file | `read_string()` | `read_double()` |\n|---|---|---|---|---|\n" + "\n".join(_rows)
        )
    except RuntimeError as _e:
        _out = mo.callout(mo.md(f"**Error parsing the file**: `{error_message(_e)}`"), kind="danger")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **Run it on your computer** (Linux, Python 3.11 to 3.14):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/MRPT/mrpt-notebooks/main/notebooks/runtime_expressions.py
    ```

    More MRPT Python examples: [mrpt_examples_py](https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_py).
    """)
    return


if __name__ == "__main__":
    app.run()
