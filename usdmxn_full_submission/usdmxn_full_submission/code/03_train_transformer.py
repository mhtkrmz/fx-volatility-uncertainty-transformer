
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from pipeline import resolve_paths, prepare_data, run_baselines, train_transformers, write_manifest
if __name__ == "__main__":
    paths = resolve_paths()
    data_ctx = prepare_data(paths)
    baseline_ctx = run_baselines(paths, data_ctx)
    train_transformers(paths, data_ctx, baseline_ctx)
    write_manifest(paths)
