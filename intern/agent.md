# Intern_Fujitsu agent guide

## Scope and working directory

- Work from the repository root: `Intern_Fujitsu/`.
- This repository compares FMQA, Standard QAOA, and FM-XY-QAOA for One-Hot
  categorical QUBOs.  Treat every conclusion as specific to the tested
  encoding, instances, p=1 circuit, grid search, classical state-vector
  simulation, and seed coverage.
- Do not describe the results as a universal quantum advantage or as universal
  final-objective superiority over FMQA.  The measured structural benefit is
  constraint preservation by the XY mixer for the tested One-Hot encoding.

## Current delivery status: intern-PC package

- The current task is to prepare the existing `intern/` directory for download
  onto the internship PC.  Do not create a second `Intern/` directory.
- `intern/` is a self-contained delivery snapshot: it retains the experiment
  hierarchy and includes source, current JSON data, integrated figures, and
  independently rendered panels so the current results can be inspected
  offline immediately.
- Code completed in this package is intended for the corresponding GitHub
  repository (`https://github.com/taiga-hayashi/QAOA_BBO.git`).  Do not push,
  tag, or publish it unless the user explicitly requests that action.
- Keep the package reproducible: source changes must be reflected both in the
  canonical `experimet/` tree and in `intern/experimet/`; then run the package
  validation described in `intern/README.md` before handing it off.

## Repository layout

```text
Intern_Fujitsu/
├── AGENT.md
├── pyproject.toml / requirements.txt      # Python 3.11+; includes openqarp
├── main.py / config.yaml / src/            # Main-project implementation
├── bb_functions/ / QAOA/ / result/ / report/
├── quarp/                                # Fujitsu/OpenQARP reference guides and reports
├── intern/                               # Internship-PC delivery package; mirrors main layout
└── experimet/                            # Experiment sources and deliverables
    ├── README.md                          # Inventory; update if status changes
    ├── py/                               # Shared runners and utilities
    │   ├── openqarp_qaoa.py               # Canonical OpenQARP p=1 circuits
    │   ├── run_openqarp_bbo_chunk.py      # Resumable 100-cycle trajectory runner
    │   ├── finalize_openqarp_bbo_results.py
    │   ├── rerun_openqarp_direct_comparison.py
    │   └── export_individual_panels.py    # Native standalone panel generator
    └── <experiment>/{py,json,png,pdf,md}/
        ├── optimization/                  # Direct comparison + 100-cycle BBO
        ├── bbo_seed_robustness/           # 3-seed, 100-cycle BBO validation
        ├── qubit_scaling/                 # N scaling and objective-value coverage
        ├── xy_qaoa_accuracy_scaling/      # Current XY-QAOA parameter-search and N scaling
        ├── xy_qaoa_normalization/         # Current XY-QAOA coefficient-normalization sensitivity
        ├── std_qaoa_sensitivity/          # Standard-QAOA penalty scan
        ├── shot_budget_sensitivity/       # Standard-QAOA shot-budget check
        ├── penalty_sensitivity/           # FMQA penalty scan
        └── adaptive_lambda_validation/    # Adaptive-penalty validation
```

Keep every experiment directory organized by extension:

- `py/`: source and plotting scripts only.
- `json/`: machine-readable raw/summary results.  Do not relabel old data as
  OpenQARP results.
- `png/` and `pdf/`: rendered integrated figures.  Native individual figures
  are under `png/panels/` and `pdf/panels/`.
- `md/`: human-readable result summaries.

Do not flatten experiment folders or replace independent panel figures with
image/PDF crops.  Each panel must be drawn into a fresh Matplotlib figure with
its own labels, margins, and legend.

`intern/` must retain the relevant top-level main-project layout (`main.py`,
`config.yaml`, `src/`, `bb_functions/`, `QAOA/`, `result/`, `report/`,
`qaoa_tutorial/`, `quarp/`, and `experimet/`).  It is not an experiments-only
folder; update its mirrored source when modifying the canonical code intended
for the internship PC.

The internship experiment workflow is a main-project extension: use
`python main.py --experiments [--fresh]` from `intern/`.  Do not replace the
ordinary `python main.py` path, which must continue to dispatch to `src/main.py`.

## QAOA implementation rules

- Use `src/qarp_backend.py` for all QAOA probability calculations.  The
  experiment-side `experimet/py/openqarp_qaoa.py` is only a compatibility
  import, so every QAOA execution uses the same OpenQARP/`qarpx` circuits.
- Standard QAOA: p=1 X mixer plus One-Hot penalty.  FM-XY-QAOA: p=1, One-Hot
  initial state and complete-graph `RXX/RYY` XY mixer without penalty.
- Preserve the existing QUBO convention, angle convention, parameter grid,
  shot count, seed handling, and objective evaluation unless the task
  explicitly requests a methodological change.
- A JSON result is an OpenQARP QAOA result only when its metadata identifies
  the OpenQARP backend.  For long BBO work, preserve every checkpoint and only
  promote a result when `next_cycle == 101`, `best_history` has 101 values,
  and `proposals` has 100 values.
- FMQA, penalty sensitivity, and adaptive-lambda validation are classical
  controls.  Recompute them when regenerating a complete comparison, but do
  not call them QARP calculations.

## Measurement boundaries and interpretation

- Compare objective values across methods at the same `N` and same instance;
  never infer a vertical trend across separately generated instances.
- In the current OpenQARP qubit-scaling data, QAOA is actually simulated only
  through `N <= 20`.  Values beyond that range must remain `null`/unmeasured,
  not be plotted or narrated as data.  FMQA coverage may extend further.
- One-Hot N=16 has `2^16 = 65,536` total basis states but only `4^4 = 256`
  feasible states.  Explain FM-XY-QAOA's feasibility benefit through that
  subspace preservation.  State explicitly that it may change under binary or
  other encodings.
- Separate: (1) measured result, (2) structural explanation, and (3)
  unverified generalization.  Do not promote a three-seed result to an
  inferential statistical claim.

## Recalculation and plotting workflow

1. Modify source under the relevant `py/` directory (or `experimet/py/` for
   shared code), then compile it.
2. Run/re-run raw calculations before plotting.  Never regenerate a figure
   from pre-migration JSON and call it an OpenQARP result.
3. Generate the integrated figure from its JSON using the experiment plotting
   script.
4. Generate all standalone panels with:

   ```sh
   MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/mplconfig-openqarp \
   XDG_CACHE_HOME=/private/tmp/xdg-cache-openqarp \
   python3 experimet/py/export_individual_panels.py
   ```

5. Render changed PDFs with `pdftoppm` and visually inspect them.  Verify
   legend/data non-overlap, no clipping, readable tick labels, and a legend in
   every standalone panel.
6. Validate JSON coverage, BBO trajectory lengths, and metadata before saying
   that a complete rerun is finished.

Useful checks:

```sh
python3 -m py_compile experimet/py/*.py experimet/*/py/*.py
pdftoppm -f 1 -l 1 -png -r 150 <figure.pdf> /private/tmp/check
```

## User-requested conventions

- Avoid plot titles by default; use axes, legends, filenames, and README/MD
  context.  Retain a title only when the user explicitly needs one or when it
  identifies a standalone panel unambiguously.
- The user wants objective-value comparisons against `N`, but only in the
  measured common range.
- For multi-result figures, supply both the integrated PDF and independently
  rendered one-figure-per-panel PDF/PNG files.
- When asked whether a calculation is complete, report actual completed raw
  trajectories and remaining work; do not infer completion from updated PDFs.
- Preserve user files and unrelated changes.  Do not delete data, checkpoints,
  or result files unless explicitly directed.

## Current implementation decisions (2026-09)

- Keep the existing `src/` layout.  Do not move analysis scripts into a new
  `workflows/` tree merely to make `src` smaller.  Instead, remove duplicated
  implementation while preserving script locations and direct execution.
- Shared code currently lives in `src/runtime.py` (configuration, seed, and
  output layout), `src/qubo.py` (matrix-to-QUBO conversion), `src/fm.py`
  (shared FM training), and `src/qarp_backend.py` (QAOA circuits).  Reuse
  those helpers; do not reintroduce local copies of the same helpers.
- `src/qaoa_solver.py` is OpenQARP-only.  It has no Qiskit/Aer dependency;
  the project requirements and lock files likewise exclude Qiskit.  Its p=1
  search is a deterministic angle grid, and p>1 uses fixed-seed candidate
  search.  Do not describe those source-level searches as the historical
  Qiskit optimizer.
- `experimet/py/openqarp_qaoa.py` is a p=1 compatibility layer that imports
  the canonical `src/qarp_backend.py`; do not implement a second QARP circuit
  there.  The source migration did not itself recalculate the existing result
  JSON/PDF/PNG snapshot.
- Qiskit-named tutorial code is historical teaching material and needs a
  separate Qiskit installation if run.  The Qiskit-named report figure
  generator is also historical/reference-only, but uses fixed plotting data.
  Neither is used by `main.py` or `main.py --experiments`.
- `intern/PYTHON_FILES.md` documents the role, execution scope, and location
  of every Python file under `intern/`.  Update it whenever a Python file is
  added, removed, renamed, or materially repurposed.
- For every new experiment in `intern/`, create a dedicated directory below
  `intern/experimet/` using the established
  `<experiment>/{py,json,png,pdf,md}/` layout.  Keep that experiment's source,
  raw data, figures, and written interpretation together; do not mix a new
  experiment's artifacts into an existing experiment directory.
- Before creating an experiment `.py` file, inspect `src/` and
  `intern/experimet/py/` for existing functionality.  Import and reuse common
  QARP circuits, QUBO conversion, FM training, plotting, checkpoint, and
  result helpers.  Create a new `.py` file only for the experiment-specific
  difference; never duplicate an existing implementation merely for a new
  experiment.  Reflect any newly reusable helper in the canonical shared
  module and update `intern/PYTHON_FILES.md`.
- A 128-GB host can target a memory-aware full-state OpenQARP N=24/N=28
  workflow only after its optimizer and result aggregation avoid dense
  all-bitstring arrays.  That larger-N implementation has not yet been run;
  do not label it as measured data.
