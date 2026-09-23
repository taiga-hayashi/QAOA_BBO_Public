# `src` architecture

`src` holds the reusable implementation behind the original FMQA/QAOA
demonstration.  It is intentionally separate from `experimet/`, which stores
the fixed-seed validation workflows and their generated artifacts.

| Module group | Responsibility |
| --- | --- |
| `bb_function.py` | Synthetic QUBO black-box objectives and feasible-pattern utilities |
| `fm.py`, `fm_to_qubo.py` | Factorization-machine surrogate, shared training loop, and QUBO conversion |
| `qubo.py` | Canonical upper-triangular matrix-to-QUBO-dictionary conversion |
| `fmqa_solver.py` | Classical FMQA/Simulated-Annealing baseline |
| `qarp_backend.py`, `qaoa_solver.py` | Canonical OpenQARP circuits and the QAOA solver API |
| `subspace_xy_qaoa_solver.py` | Constraint-preserving reduced-subspace XY-QAOA prototype |
| `runtime.py` | Validated YAML configuration, deterministic seeding, project paths, and artifact directories |
| `main.py` | Small end-to-end FMQA/QAOA demonstration |
| `*_benchmark.py`, `*_experiment.py`, `compare_*.py` | Reproducible comparison and analysis workflows |

## Entrypoints

Run the original, small FMQA/QAOA demonstration from the project root:

```sh
python main.py
```

Run the current fixed-seed OpenQARP validation suite:

```sh
python main.py --experiments
```

The experimental QAOA results are implemented in
`src/qarp_backend.py` and invoked by scripts below `experimet/<experiment>/py/`.
The experiment helper is a compatibility import only; all QAOA circuits use
OpenQARP/`qarpx`.

## Conventions

- Read the original demonstration parameters from `config.yaml` through
  `MainRunConfig`; reject invalid values before a run starts.
- Call `seed_everything(seed)` before stochastic work so Python, NumPy, and
  PyTorch use the same seed.
- Store generated artifacts under `result/{txt,json,csv,png,pdf}/`.
- Keep public solver return shapes stable.  New solver options should be
  explicit keyword arguments and validate unsupported values immediately.
- Use `matrix_to_qubo_dict` and `train_factorization_machine` rather than
  copying conversion or Adam/MSE training loops into an experiment script.

Direct execution (`python src/main.py`) and package execution
(`python -m src.main`) are both supported for the primary workflow.
