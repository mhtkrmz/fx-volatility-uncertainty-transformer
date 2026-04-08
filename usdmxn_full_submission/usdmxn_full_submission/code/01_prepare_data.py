
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from pipeline import resolve_paths, prepare_data, write_manifest
if __name__ == "__main__":
    paths = resolve_paths()
    prepare_data(paths)
    write_manifest(paths)
