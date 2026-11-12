# MRPT notebooks

Interactive [marimo](https://marimo.io) notebooks showing the
[MRPT](https://www.mrpt.org/) Python API, installed from PyPI with
`pip install mrpt3`.

| Notebook | Modules | Open online |
|---|---|---|
| [ICP scan matching](notebooks/icp_scan_matching.py) | `mrpt.poses`, `mrpt.maps`, `mrpt.slam` | [molab](https://molab.marimo.io/github/MRPT/mrpt-notebooks/blob/main/notebooks/icp_scan_matching.py) |

The molab link opens a static preview with the saved outputs. To move the
sliders, run it on a server from the preview menu.

## Run locally

Requires Linux (x86_64 or aarch64) and Python 3.11 to 3.14, the platforms of
the `mrpt3` wheels. With [uv](https://docs.astral.sh/uv/):

```bash
uvx marimo edit --sandbox notebooks/icp_scan_matching.py
```

`--sandbox` installs the dependencies listed at the top of each notebook in a
temporary environment.

## Saved outputs

The previews show the outputs stored in `notebooks/__marimo__/session/`.
After changing a notebook, refresh them with:

```bash
uvx marimo export session --sandbox notebooks/
```
