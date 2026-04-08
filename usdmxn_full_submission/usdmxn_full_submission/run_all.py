
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from pipeline import resolve_paths, run_all

if __name__ == "__main__":
    paths = resolve_paths()
    run_all(paths, compile_after=False)
