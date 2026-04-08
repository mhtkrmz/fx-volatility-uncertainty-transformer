
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from pipeline import resolve_paths, prepare_data, run_baselines, train_transformers, build_tables, plot_core_figures, plot_additional_figures, compile_paper, sync_figures_to_paper, write_manifest
if __name__ == "__main__":
    paths = resolve_paths()
    data_ctx = prepare_data(paths)
    baseline_ctx = run_baselines(paths, data_ctx)
    transformer_ctx = train_transformers(paths, data_ctx, baseline_ctx)
    table_ctx = build_tables(paths, data_ctx, baseline_ctx, transformer_ctx)
    plot_core_figures(paths, data_ctx, baseline_ctx, transformer_ctx, table_ctx)
    plot_additional_figures(paths, data_ctx, table_ctx)
    sync_figures_to_paper(paths)
    compile_paper(paths)
    write_manifest(paths)
