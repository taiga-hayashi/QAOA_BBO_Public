"""Run preserved intern_1009 scripts using their canonical shared dependencies."""
import importlib
import os
from pathlib import Path
import runpy
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'src'
sys.path[:0]=[str(SRC),str(ROOT)]
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'intern1009-public-mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'intern1009-public-cache'))
# The public repository also contains older, flat modules at its root. Preload
# canonical modules so preserved runners cannot accidentally import those files.
for name in ['bb_function','fm','fm_to_qubo','fmqa_solver','qarp_backend','qaoa_solver']:
    module=importlib.import_module(name)
    assert Path(module.__file__).resolve().parent==SRC

args=sys.argv[1:]
check=bool(args and args[0]=='--check-imports')
if check:args=args[1:]
if not args:
    if check:
        print('Canonical shared imports: passed')
        sys.exit(0)
    raise SystemExit('Usage: python intern_1009/py/run_public.py [--check-imports] intern_1009/<problem>/<experiment>/py/<script>.py [arguments]')
target=(ROOT/args[0]).resolve()
if not target.is_relative_to(ROOT/'intern_1009') or target.suffix!='.py':
    raise SystemExit('Target must be a .py file inside intern_1009')
sys.path.insert(0,str(target.parent))
sys.argv=[str(target),*args[1:]]
runpy.run_path(str(target),run_name='public_import_check' if check else '__main__')
if check:print('Preserved runner imports: passed')
