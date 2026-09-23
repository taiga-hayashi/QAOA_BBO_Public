# Intern-PC experiment package

This folder is the downloadable, self-contained package for the internship
PC.  Its top-level code layout follows the current main project; `experimet/`
is retained as the validation/reproducibility subtree.  It includes the current
result JSON, integrated PDF/PNG figures, and standalone native panel figures.
The supplied sources regenerate those artifacts with the same fixed seeds and
settings.

For the role and execution scope of every included Python file, see
[`PYTHON_FILES.md`](PYTHON_FILES.md).

## Contents

```text
intern/
├── README.md
├── agent.md                         # Current operating rules and delivery status
├── pyproject.toml
├── requirements.txt
├── main.py / config.yaml             # Main-project entry point and configuration
├── src/                              # Main FMQA/QAOA implementation
├── bb_functions/                     # Black-box-function documentation
├── QAOA/                             # QAOA guide
├── result/                           # Main-project outputs, grouped by extension
├── report/                           # Main-project reports and figure generators
├── qaoa_tutorial/                    # Tutorial material
├── quarp/                            # OpenQARP reference material
├── run_all_experiments.py             # implementation behind main.py --experiments
└── experimet/
    ├── py/                         # shared OpenQARP and BBO utilities
    └── <experiment>/{py,json,png,pdf,md}/
```

Each experiment directory keeps sources in `py/`, data in `json/`, rendered
figures in `png/` and `pdf/`, and summaries in `md/`.  Individual native
figures are in `png/panels/` and `pdf/panels/`; they are not crops of a
multi-panel image.

The directory is intentionally independent of the original machine: it does
not contain `.git/`, `.venv/`, Python bytecode, or local cache directories.

## Setup

Use Python 3.11 or newer.  From this `intern/` directory:

```sh
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`openqarp` is required for the QAOA calculations.  The included result files
are the current validated snapshot, so they can be viewed without rerunning
the expensive BBO trajectories.

## Reproduce the current outputs

```sh
python main.py --experiments --fresh
```

`main.py --experiments` extends the normal main-project entry point.  Without
`--experiments`, `python main.py` runs the original FMQA/QAOA workflow through
`src/main.py` exactly as before.  `--fresh` removes only local resumable BBO checkpoints and reruns the complete
fixed-seed calculation.  It then regenerates integrated plots and all native
standalone panels.  A full BBO run is intentionally long; checkpoints make it
safe to restart after interruption.  Without `--fresh`, the script resumes
unfinished trajectories and rebuilds outputs from available results.

For a quick source check:

```sh
python -m py_compile experimet/py/*.py experimet/*/py/*.py
```

## Scientific scope

The QAOA circuits use OpenQARP/`qarpx` state-vector simulation with p=1.
Standard QAOA uses an X mixer plus One-Hot penalty; FM-XY-QAOA uses a One-Hot
initial state and an XY mixer.  Claims are specific to the tested One-Hot
instances and should not be generalized to other encodings or hardware.

In the current QAOA scaling data, only `N <= 20` is an actual OpenQARP
simulation; larger QAOA entries are explicitly unmeasured (`null`).

## GitHub handoff

The intended repository is `taiga-hayashi/QAOA_BBO`.  Commit source,
configuration, reproducibility instructions, and validated lightweight result
artifacts as directed by the user.  Do not publish local virtual environments,
cache directories, or temporary PDF renderings.
