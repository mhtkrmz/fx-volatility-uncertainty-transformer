
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from pipeline import resolve_paths, prepare_data, run_baselines, write_manifest
if __name__ == "__main__":
    paths = resolve_paths()
    data_ctx = prepare_data(paths)
    run_baselines(paths, data_ctx)
    write_manifest(paths)
