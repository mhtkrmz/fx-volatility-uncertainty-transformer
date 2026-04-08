
"""End-to-end reproducibility pipeline for the USD/MXN volatility paper.

This module covers the full path from raw FX panel data to:
1. cleaned and engineered datasets,
2. baseline model fitting,
3. Transformer training and prediction,
4. tables and figures,
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.stattools import acf as sm_acf
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(8)


@dataclass(frozen=True)
class ExperimentConfig:
    countries: Tuple[str, ...] = (
        "Mexico",
        "Brazil",
        "Canada",
        "Euro",
        "South Africa",
        "Japan",
        "South Korea",
        "China",
    )
    start_date: str = "2000-01-01"
    end_date: str = "2026-03-27"
    train_end: str = "2015-12-31"
    val_end: str = "2019-12-31"
    eps: float = 1e-8
    lookback: int = 40
    patch_size: int = 5
    transformer_univariate_seed: int = 7
    transformer_ensemble_seeds: Tuple[int, ...] = (11, 17, 23)
    transformer_epochs: int = 14
    transformer_patience: int = 4
    transformer_batch_size: int = 512
    transformer_lr: float = 6e-4


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_data: Path
    code: Path
    outputs: Path
    processed: Path
    models: Path
    predictions: Path
    tables: Path
    figures: Path
    reports: Path
    paper: Path
    paper_figures: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        outputs = root / "outputs"
        return cls(
            root=root,
            raw_data=root / "data" / "raw" / "daily_fx.csv",
            code=root / "code",
            outputs=outputs,
            processed=outputs / "processed",
            models=outputs / "models",
            predictions=outputs / "predictions",
            tables=outputs / "tables",
            figures=outputs / "figures",
            reports=outputs / "reports",
            paper=root / "paper",
            paper_figures=root / "paper" / "figures",
        )

    def ensure(self) -> None:
        for path in (
            self.code,
            self.processed,
            self.models,
            self.predictions,
            self.tables,
            self.figures,
            self.reports,
            self.paper,
            self.paper_figures,
        ):
            path.mkdir(parents=True, exist_ok=True)


CONFIG = ExperimentConfig()


def resolve_paths() -> ProjectPaths:
    root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    return paths


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _round_dict(values: Mapping[str, float], digits: int = 6) -> Dict[str, float]:
    return {key: round(float(value), digits) for key, value in values.items()}


def _json_dump(obj: object, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def load_raw_data(paths: ProjectPaths) -> pd.DataFrame:
    if not paths.raw_data.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {paths.raw_data}. "
            "Expected the daily FX panel CSV inside data/raw/."
        )
    df = pd.read_csv(paths.raw_data, parse_dates=["Date"])
    if "Exchange rate" not in df.columns and "Value" in df.columns:
        df = df.rename(columns={"Value": "Exchange rate"})
    required = {"Date", "Country", "Exchange rate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    return df


def prepare_data(
    paths: ProjectPaths,
    config: ExperimentConfig = CONFIG,
    save_outputs: bool = True,
) -> Dict[str, object]:
    df = load_raw_data(paths)
    panel = (
        df[df["Country"].isin(config.countries)]
        .pivot(index="Date", columns="Country", values="Exchange rate")
        .sort_index()
    )
    panel = panel[(panel.index >= config.start_date) & (panel.index <= config.end_date)].dropna()

    log_levels = np.log(panel)
    rets = log_levels.diff().dropna()
    rv = rets**2
    log_rv = np.log(rv + config.eps)
    y = log_rv["Mexico"]

    feat = pd.DataFrame(index=y.index)
    for country in config.countries:
        feat[f"{country}_ret"] = rets[country]
        feat[f"{country}_logrv"] = log_rv[country]
    feat["dow"] = feat.index.dayofweek
    for d in range(5):
        feat[f"dow_{d}"] = (feat["dow"] == d).astype(float)
    feat = feat.drop(columns=["dow"])
    feat["y_w"] = y.rolling(5).mean().shift(1)
    feat["y_m"] = y.rolling(22).mean().shift(1)
    feat = feat.dropna()

    base = pd.DataFrame(
        {
            "y": y,
            "y_lag1": y.shift(1),
            "y_w": y.rolling(5).mean().shift(1),
            "y_m": y.rolling(22).mean().shift(1),
        }
    ).dropna()

    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)

    summary = {
        "countries": list(config.countries),
        "start_date": str(panel.index.min().date()),
        "end_date": str(panel.index.max().date()),
        "aligned_panel_observations": int(len(panel)),
        "return_observations": int(len(rets)),
        "feature_columns": list(feat.columns),
        "train_rows": int((base.index <= train_end).sum()),
        "validation_rows": int(((base.index > train_end) & (base.index <= val_end)).sum()),
        "test_rows": int((base.index > val_end).sum()),
        "config": asdict(config),
    }

    if save_outputs:
        panel.to_csv(paths.processed / "panel.csv.gz", compression="gzip")
        rets.to_csv(paths.processed / "returns.csv.gz", compression="gzip")
        log_rv.to_csv(paths.processed / "log_rv.csv.gz", compression="gzip")
        feat.to_csv(paths.processed / "feature_matrix.csv.gz", compression="gzip")
        base.to_csv(paths.processed / "har_frame.csv.gz", compression="gzip")
        _json_dump(summary, paths.processed / "data_summary.json")

    return {
        "panel": panel,
        "returns": rets,
        "rv": rv,
        "log_rv": log_rv,
        "target": y,
        "features": feat,
        "base": base,
        "summary": summary,
    }


def gaussian_nll_np(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.clip(np.asarray(sigma), 1e-6, None)
    y_true = np.asarray(y_true)
    mu = np.asarray(mu)
    return 0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y_true - mu) ** 2) / (sigma**2)


def gaussian_crps_np(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.clip(np.asarray(sigma), 1e-6, None)
    z = (np.asarray(y_true) - np.asarray(mu)) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    width = upper - lower
    below = (y_true < lower).astype(float)
    above = (y_true > upper).astype(float)
    return width + (2 / alpha) * (lower - y_true) * below + (2 / alpha) * (y_true - upper) * above


def qlike(rv_true: np.ndarray, rv_pred: np.ndarray) -> np.ndarray:
    rv_true = np.clip(np.asarray(rv_true), 1e-12, None)
    rv_pred = np.clip(np.asarray(rv_pred), 1e-12, None)
    ratio = rv_true / rv_pred
    return ratio - np.log(ratio) - 1


def summarize_density(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: float = CONFIG.eps,
    alpha: float = 0.1,
) -> Dict[str, float]:
    sigma = np.clip(np.asarray(sigma), 1e-6, None)
    y_true = np.asarray(y_true)
    mu = np.asarray(mu)
    lower = mu + norm.ppf(alpha / 2) * sigma
    upper = mu + norm.ppf(1 - alpha / 2) * sigma
    rv_true = np.exp(y_true) - eps
    rv_pred = np.exp(mu + 0.5 * sigma**2) - eps
    return {
        "NLL": float(np.mean(gaussian_nll_np(y_true, mu, sigma))),
        "CRPS": float(np.mean(gaussian_crps_np(y_true, mu, sigma))),
        "RMSE_y": float(np.sqrt(np.mean((y_true - mu) ** 2))),
        "MAE_y": float(np.mean(np.abs(y_true - mu))),
        "Cov90": float(np.mean((y_true >= lower) & (y_true <= upper))),
        "IS90": float(np.mean(interval_score(y_true, lower, upper, alpha=0.1))),
        "QLIKE": float(np.mean(qlike(rv_true, rv_pred))),
    }


def fit_ols_gaussian(df_train: pd.DataFrame, features: Sequence[str]) -> Tuple[np.ndarray, float]:
    X = np.column_stack([np.ones(len(df_train)), df_train[list(features)].values])
    yv = df_train["y"].values
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ beta
    sigma = resid.std(ddof=X.shape[1])
    return beta, float(sigma)


def predict_ols_gaussian(
    df_eval: pd.DataFrame,
    beta: np.ndarray,
    sigma: float,
    features: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(len(df_eval)), df_eval[list(features)].values])
    mu = X @ beta
    sig = np.full(len(df_eval), sigma)
    return mu, sig


def fit_quantile_models(
    df_train: pd.DataFrame,
    features: Sequence[str],
    taus: Sequence[float] = (0.05, 0.5, 0.95),
) -> Dict[float, object]:
    X = pd.DataFrame({"const": 1.0}, index=df_train.index)
    for feature in features:
        X[feature] = df_train[feature]
    models = {}
    for tau in taus:
        models[float(tau)] = QuantReg(df_train["y"], X).fit(q=tau, max_iter=5000)
    return models


def predict_quantile_models(
    df_eval: pd.DataFrame,
    models: Mapping[float, object],
    features: Sequence[str],
) -> Dict[float, np.ndarray]:
    X = pd.DataFrame({"const": 1.0}, index=df_eval.index)
    for feature in features:
        X[feature] = df_eval[feature]
    return {tau: np.asarray(model.predict(X)) for tau, model in models.items()}


def garch_recursion(
    resid: np.ndarray,
    params: Sequence[float],
    model: str = "garch",
    h0: Optional[float] = None,
) -> np.ndarray:
    resid = np.asarray(resid, dtype=float)
    n = len(resid)
    if h0 is None:
        h0 = float(np.var(resid))
    h = np.empty(n)
    if model == "garch":
        omega, alpha, beta = params
        h_prev = max(h0, 1e-8)
        for t in range(n):
            if t == 0:
                h[t] = h_prev
            else:
                h_prev = omega + alpha * resid[t - 1] ** 2 + beta * h_prev
                h_prev = max(h_prev, 1e-8)
                h[t] = h_prev
    elif model == "gjr":
        omega, alpha, gamma, beta = params
        h_prev = max(h0, 1e-8)
        for t in range(n):
            if t == 0:
                h[t] = h_prev
            else:
                e_prev = resid[t - 1]
                h_prev = omega + alpha * e_prev**2 + gamma * (e_prev < 0) * e_prev**2 + beta * h_prev
                h_prev = max(h_prev, 1e-8)
                h[t] = h_prev
    elif model == "egarch":
        omega, alpha, gamma, beta = params
        expected_abs_gaussian = np.sqrt(2 / np.pi)
        logh_prev = np.log(max(h0, 1e-8))
        for t in range(n):
            if t == 0:
                h[t] = float(np.exp(logh_prev))
            else:
                z_prev = resid[t - 1] / np.sqrt(max(np.exp(logh_prev), 1e-8))
                logh_prev = omega + beta * logh_prev + alpha * (abs(z_prev) - expected_abs_gaussian) + gamma * z_prev
                logh_prev = np.clip(logh_prev, -12, 12)
                h[t] = float(np.exp(logh_prev))
    else:
        raise ValueError(f"Unknown GARCH model type: {model}")
    return h


def garch_nll(
    params: Sequence[float],
    resid: np.ndarray,
    model: str = "garch",
    h0: Optional[float] = None,
) -> float:
    if model == "garch":
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
    elif model == "gjr":
        omega, alpha, gamma, beta = params
        if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0 or alpha + beta + 0.5 * gamma >= 0.999:
            return 1e12
    elif model == "egarch":
        omega, alpha, gamma, beta = params
        if not (-0.999 < beta < 0.999):
            return 1e12

    h = garch_recursion(resid, params, model=model, h0=h0)
    ll = gaussian_nll_np(resid, np.zeros_like(resid), np.sqrt(h))
    if not np.isfinite(ll).all():
        return 1e12
    return float(np.sum(ll))


def fit_garch_resid(resid: np.ndarray, model: str = "garch") -> Dict[str, object]:
    resid = np.asarray(resid, dtype=float)
    variance0 = float(np.var(resid))
    h0 = max(variance0, 1e-6)
    if model == "garch":
        x0 = np.array([0.03 * variance0, 0.05, 0.90])
        bounds = [(1e-8, None), (1e-8, 0.999), (1e-8, 0.999)]
    elif model == "gjr":
        x0 = np.array([0.03 * variance0, 0.05, 0.05, 0.85])
        bounds = [(1e-8, None), (1e-8, 0.999), (1e-8, 0.999), (1e-8, 0.999)]
    elif model == "egarch":
        x0 = np.array([0.05 * np.log(variance0), 0.05, -0.05, 0.90])
        bounds = [(-10, 10), (-5, 5), (-5, 5), (-0.999, 0.999)]
    else:
        raise ValueError(f"Unknown GARCH model type: {model}")

    best = None
    for scale in (0.5, 1.0, 2.0):
        xstart = x0.copy()
        xstart[0] *= scale
        result = minimize(
            garch_nll,
            xstart,
            args=(resid, model, h0),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    return {"params": [float(x) for x in best.x], "h0": float(h0), "model": model}


def run_baselines(
    paths: ProjectPaths,
    data_ctx: Dict[str, object],
    config: ExperimentConfig = CONFIG,
) -> Dict[str, object]:
    base = data_ctx["base"]
    target = data_ctx["target"]
    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)

    train = base.loc[:train_end].copy()
    val = base.loc[(base.index > train_end) & (base.index <= val_end)].copy()
    test = base.loc[base.index > val_end].copy()

    har_features = ["y_lag1", "y_w", "y_m"]
    beta_har, sigma_har = fit_ols_gaussian(train, har_features)
    full = base.copy()
    mu_full, _ = predict_ols_gaussian(full, beta_har, sigma_har, har_features)
    resid_full = full["y"].values - mu_full

    train_len, val_len, test_len = len(train), len(val), len(test)
    mu_test = mu_full[train_len + val_len :]
    y_test = test["y"].values

    density_metrics: Dict[str, Dict[str, float]] = {}
    density_metrics["HAR-style Gaussian"] = summarize_density(y_test, mu_test, np.full(test_len, sigma_har))

    sigma_lookup: Dict[str, np.ndarray] = {"HAR-style Gaussian": np.full(test_len, sigma_har)}
    garch_params: Dict[str, Dict[str, object]] = {}
    for pretty_name, key in (("GARCH", "garch"), ("GJR-GARCH", "gjr"), ("EGARCH", "egarch")):
        fitted = fit_garch_resid(resid_full[:train_len], model=key)
        garch_params[pretty_name] = fitted
        h_full = garch_recursion(resid_full, fitted["params"], model=key, h0=fitted["h0"])
        sigma_lookup[pretty_name] = np.sqrt(h_full[train_len + val_len :])
        density_metrics[pretty_name] = summarize_density(y_test, mu_test, sigma_lookup[pretty_name])

    quantile_models = fit_quantile_models(train, har_features)
    quantile_preds = predict_quantile_models(test, quantile_models, har_features)
    quantile_metrics = {
        "Cov90": float(np.mean((y_test >= quantile_preds[0.05]) & (y_test <= quantile_preds[0.95]))),
        "IS90": float(np.mean(interval_score(y_test, quantile_preds[0.05], quantile_preds[0.95], alpha=0.1))),
        "RMSE_0.50": float(np.sqrt(np.mean((y_test - quantile_preds[0.5]) ** 2))),
    }
    for tau, pred in quantile_preds.items():
        errors = y_test - pred
        quantile_metrics[f"Pinball_{tau:.2f}"] = float(np.mean(np.maximum(tau * errors, (tau - 1.0) * errors)))

    baseline_predictions = pd.DataFrame(
        {
            "Date": pd.to_datetime(test.index),
            "y_true": y_test,
            "har_mu": mu_test,
            "har_sigma": sigma_lookup["HAR-style Gaussian"],
            "garch_sigma": sigma_lookup["GARCH"],
            "gjr_sigma": sigma_lookup["GJR-GARCH"],
            "egarch_sigma": sigma_lookup["EGARCH"],
            "q05": quantile_preds[0.05],
            "q50": quantile_preds[0.5],
            "q95": quantile_preds[0.95],
        }
    )

    baseline_predictions.to_csv(paths.predictions / "baseline_predictions.csv", index=False)
    pd.DataFrame(density_metrics).T.to_csv(paths.tables / "baseline_density_metrics.csv")
    pd.DataFrame({"metric": list(quantile_metrics.keys()), "value": list(quantile_metrics.values())}).to_csv(
        paths.tables / "quantile_metrics_long.csv", index=False
    )
    _json_dump({"beta": [float(x) for x in beta_har], "sigma": float(sigma_har)}, paths.models / "har_gaussian.json")
    _json_dump(garch_params, paths.models / "garch_family_parameters.json")
    _json_dump({"density_metrics": density_metrics, "quantile_metrics": quantile_metrics}, paths.reports / "baseline_summary.json")

    return {
        "train": train,
        "val": val,
        "test": test,
        "mu_test": mu_test,
        "y_test": y_test,
        "har_features": har_features,
        "density_metrics": density_metrics,
        "quantile_metrics": quantile_metrics,
        "quantile_predictions": quantile_preds,
        "baseline_predictions": baseline_predictions,
        "sigma_lookup": sigma_lookup,
        "har_beta": beta_har,
        "har_sigma": sigma_har,
        "resid_full": resid_full,
    }


def make_sequences(
    data_ctx: Dict[str, object],
    config: ExperimentConfig = CONFIG,
    save_outputs: bool = True,
    paths: Optional[ProjectPaths] = None,
) -> Dict[str, object]:
    feat = data_ctx["features"]
    target = data_ctx["target"]

    common_idx = feat.index.intersection(target.index)
    feat = feat.loc[common_idx]
    target = target.loc[common_idx]

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    dates_list: List[pd.Timestamp] = []

    for i in range(config.lookback, len(common_idx)):
        xseq = feat.iloc[i - config.lookback : i].values.astype(np.float32)
        yt = float(target.iloc[i])
        if np.isnan(xseq).any() or np.isnan(yt):
            continue
        X_list.append(xseq)
        y_list.append(yt)
        dates_list.append(common_idx[i])

    X_seq = np.stack(X_list)
    y_seq = np.array(y_list, dtype=np.float32)
    dates_seq = pd.to_datetime(np.array(dates_list))
    feature_names = list(feat.columns)
    univariate_features = [f for f in feature_names if f.startswith("Mexico_")] + ["y_w", "y_m"] + [f"dow_{d}" for d in range(5)]

    if save_outputs and paths is not None:
        seq_meta = pd.DataFrame({"Date": dates_seq})
        seq_meta.to_csv(paths.processed / "sequence_dates.csv", index=False)
        _json_dump(
            {
                "n_sequences": int(len(X_seq)),
                "lookback": int(config.lookback),
                "feature_names": feature_names,
                "univariate_features": univariate_features,
                "array_shape": [int(x) for x in X_seq.shape],
            },
            paths.processed / "sequence_summary.json",
        )

    return {
        "X_seq": X_seq,
        "y_seq": y_seq,
        "dates_seq": dates_seq,
        "feature_names": feature_names,
        "univariate_features": univariate_features,
    }


def prepare_dataset(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    dates_arr: np.ndarray,
    feature_names: Sequence[str],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    subset: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    if subset is None:
        feat_idx = np.arange(len(feature_names))
        subset = list(feature_names)
    else:
        feat_idx = np.array([feature_names.index(f) for f in subset])

    Xs = X_arr[:, :, feat_idx].astype(np.float32)
    train_mask = dates_arr <= train_end
    val_mask = (dates_arr > train_end) & (dates_arr <= val_end)
    test_mask = dates_arr > val_end

    mu_x = Xs[train_mask].reshape(-1, Xs.shape[-1]).mean(axis=0)
    std_x = Xs[train_mask].reshape(-1, Xs.shape[-1]).std(axis=0)
    std_x = np.where(std_x < 1e-6, 1.0, std_x)
    Xn = (Xs - mu_x[None, None, :]) / std_x[None, None, :]

    mu_y = float(y_arr[train_mask].mean())
    std_y = float(y_arr[train_mask].std())
    yn = (y_arr - mu_y) / std_y

    return {
        "X_train": torch.tensor(Xn[train_mask], dtype=torch.float32),
        "y_train": torch.tensor(yn[train_mask], dtype=torch.float32),
        "X_val": torch.tensor(Xn[val_mask], dtype=torch.float32),
        "y_val": torch.tensor(yn[val_mask], dtype=torch.float32),
        "X_test": torch.tensor(Xn[test_mask], dtype=torch.float32),
        "y_test": torch.tensor(y_arr[test_mask], dtype=torch.float32),
        "dates_test": dates_arr[test_mask],
        "mu_y": mu_y,
        "std_y": std_y,
        "features": list(subset),
        "mu_x": mu_x.tolist(),
        "std_x": std_x.tolist(),
    }


class TinyPatchProbTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lookback: int = CONFIG.lookback,
        patch_size: int = CONFIG.patch_size,
        d_model: int = 16,
        nhead: int = 4,
        num_layers: int = 1,
        dim_ff: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert lookback % patch_size == 0, "lookback must be divisible by patch size"
        self.n_patches = lookback // patch_size
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(input_dim * patch_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.mu = nn.Linear(d_model, 1)
        self.logvar = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length, features = x.shape
        x = x.reshape(batch, self.n_patches, self.patch_size * features)
        x = self.patch_proj(x) + self.pos_emb
        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        z = self.encoder(x)
        h = self.norm(z[:, 0])
        mu = self.mu(h).squeeze(-1)
        logvar = self.logvar(h).squeeze(-1).clamp(-8, 6)
        return mu, logvar


def nll_torch(y_std: torch.Tensor, mu_std: torch.Tensor, logvar_std: torch.Tensor) -> torch.Tensor:
    return 0.5 * (math.log(2 * math.pi) + logvar_std + (y_std - mu_std) ** 2 / torch.exp(logvar_std))


def train_tiny(
    data: Dict[str, object],
    seed: int,
    config: ExperimentConfig = CONFIG,
) -> Tuple[TinyPatchProbTransformer, pd.DataFrame, float, int]:
    set_seed(seed)
    model = TinyPatchProbTransformer(
        input_dim=data["X_train"].shape[-1],
        lookback=config.lookback,
        patch_size=config.patch_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.transformer_lr, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(data["X_train"], data["y_train"]), batch_size=config.transformer_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(data["X_val"], data["y_val"]), batch_size=config.transformer_batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    history: List[Dict[str, float]] = []
    bad_epochs = 0

    for epoch in range(1, config.transformer_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            mu, lv = model(xb)
            loss = nll_torch(yb, mu, lv).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                mu, lv = model(xb)
                val_losses.append(float(nll_torch(yb, mu, lv).mean().item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_nll_std": train_loss, "val_nll_std": val_loss})

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.transformer_patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), float(best_val), int(best_epoch)


@torch.no_grad()
def predict_tiny(model: TinyPatchProbTransformer, data: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(TensorDataset(data["X_test"]), batch_size=CONFIG.transformer_batch_size, shuffle=False)
    mus: List[np.ndarray] = []
    sigs: List[np.ndarray] = []
    for (xb,) in loader:
        mu_std, logvar_std = model(xb)
        sigma_std = torch.exp(0.5 * logvar_std)
        mus.append((mu_std * data["std_y"] + data["mu_y"]).numpy())
        sigs.append((sigma_std * data["std_y"]).numpy())
    return np.concatenate(mus), np.concatenate(sigs)


def train_transformers(
    paths: ProjectPaths,
    data_ctx: Dict[str, object],
    baseline_ctx: Dict[str, object],
    config: ExperimentConfig = CONFIG,
) -> Dict[str, object]:
    sequence_ctx = make_sequences(data_ctx, config=config, save_outputs=True, paths=paths)
    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)

    data_multi = prepare_dataset(
        sequence_ctx["X_seq"],
        sequence_ctx["y_seq"],
        sequence_ctx["dates_seq"],
        sequence_ctx["feature_names"],
        train_end,
        val_end,
        subset=None,
    )
    data_uni = prepare_dataset(
        sequence_ctx["X_seq"],
        sequence_ctx["y_seq"],
        sequence_ctx["dates_seq"],
        sequence_ctx["feature_names"],
        train_end,
        val_end,
        subset=sequence_ctx["univariate_features"],
    )

    _json_dump(
        {
            "multivariate_features": data_multi["features"],
            "univariate_features": data_uni["features"],
            "multivariate_scaling": {"mu_y": data_multi["mu_y"], "std_y": data_multi["std_y"]},
            "univariate_scaling": {"mu_y": data_uni["mu_y"], "std_y": data_uni["std_y"]},
            "multivariate_shapes": {
                "X_train": list(data_multi["X_train"].shape),
                "X_val": list(data_multi["X_val"].shape),
                "X_test": list(data_multi["X_test"].shape),
            },
            "univariate_shapes": {
                "X_train": list(data_uni["X_train"].shape),
                "X_val": list(data_uni["X_val"].shape),
                "X_test": list(data_uni["X_test"].shape),
            },
        },
        paths.models / "transformer_dataset_metadata.json",
    )

    uni_model, uni_hist, uni_best_val, uni_best_epoch = train_tiny(
        data_uni, seed=config.transformer_univariate_seed, config=config
    )
    uni_mu, uni_sig = predict_tiny(uni_model, data_uni)
    uni_metrics = summarize_density(data_uni["y_test"].numpy(), uni_mu, uni_sig)
    uni_hist.to_csv(paths.models / "univariate_history.csv", index=False)

    member_preds = []
    member_histories = []
    member_summaries = []
    for seed in config.transformer_ensemble_seeds:
        model, hist, best_val, best_epoch = train_tiny(data_multi, seed=seed, config=config)
        mu_m, sig_m = predict_tiny(model, data_multi)
        member_df = pd.DataFrame({"Date": pd.to_datetime(data_multi["dates_test"]), "mu": mu_m, "sigma": sig_m})
        member_df.to_csv(paths.models / f"ensemble_member_seed_{seed}.csv", index=False)
        member_preds.append((mu_m, sig_m))
        member_histories.append(hist.assign(seed=seed))
        member_summaries.append({"seed": int(seed), "best_val_std_nll": float(best_val), "best_epoch": int(best_epoch)})

    mu_stack = np.vstack([p[0] for p in member_preds])
    sig_stack = np.vstack([p[1] for p in member_preds])
    mu_ens = mu_stack.mean(axis=0)
    aleatoric_var = (sig_stack**2).mean(axis=0)
    epistemic_var = mu_stack.var(axis=0, ddof=0)
    ens_sigma = np.sqrt(aleatoric_var + epistemic_var)

    ensemble_metrics = summarize_density(data_multi["y_test"].numpy(), mu_ens, ens_sigma)
    all_hist = pd.concat(member_histories, ignore_index=True)
    all_hist.to_csv(paths.models / "ensemble_histories.csv", index=False)

    predictions = pd.DataFrame(
        {
            "Date": pd.to_datetime(data_multi["dates_test"]),
            "y_true": data_multi["y_test"].numpy(),
            "uni_mu": uni_mu,
            "uni_sigma": uni_sig,
            "ens_mu": mu_ens,
            "ens_sigma": ens_sigma,
            "aleatoric_sd": np.sqrt(aleatoric_var),
            "epistemic_sd": np.sqrt(epistemic_var),
        }
    )
    predictions.to_csv(paths.predictions / "transformer_predictions.csv", index=False)

    _json_dump(
        {
            "univariate_metrics": uni_metrics,
            "ensemble_metrics": ensemble_metrics,
            "univariate_best_epoch": uni_best_epoch,
            "univariate_best_val_std_nll": uni_best_val,
            "ensemble_members": member_summaries,
        },
        paths.reports / "transformer_summary.json",
    )

    return {
        "sequence_ctx": sequence_ctx,
        "data_multi": data_multi,
        "data_uni": data_uni,
        "uni_metrics": uni_metrics,
        "ensemble_metrics": ensemble_metrics,
        "predictions": predictions,
        "ensemble_histories": all_hist,
        "member_summaries": member_summaries,
    }


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray) -> Dict[str, float]:
    d = np.asarray(loss_a) - np.asarray(loss_b)
    stat = d.mean() / np.sqrt(d.var(ddof=1) / len(d))
    p_value = 2 * (1 - norm.cdf(abs(stat)))
    return {"stat": float(stat), "p_value": float(p_value)}


def build_tables(
    paths: ProjectPaths,
    data_ctx: Dict[str, object],
    baseline_ctx: Dict[str, object],
    transformer_ctx: Dict[str, object],
    config: ExperimentConfig = CONFIG,
) -> Dict[str, object]:
    density_table = pd.DataFrame.from_dict(
        {
            "HAR-style Gaussian": baseline_ctx["density_metrics"]["HAR-style Gaussian"],
            "GARCH": baseline_ctx["density_metrics"]["GARCH"],
            "GJR-GARCH": baseline_ctx["density_metrics"]["GJR-GARCH"],
            "EGARCH": baseline_ctx["density_metrics"]["EGARCH"],
            "Transformer (univariate)": transformer_ctx["uni_metrics"],
            "Transformer ensemble": transformer_ctx["ensemble_metrics"],
        },
        orient="index",
    )[["NLL", "CRPS", "RMSE_y", "MAE_y", "Cov90", "IS90", "QLIKE"]]
    density_table.to_csv(paths.tables / "density_metrics.csv")

    predictions = transformer_ctx["predictions"].copy()
    baseline_preds = baseline_ctx["baseline_predictions"].copy()
    merged = baseline_preds.merge(predictions, on=["Date", "y_true"], how="left")
    merged.to_csv(paths.predictions / "all_model_predictions.csv", index=False)

    loss_ens = gaussian_nll_np(merged["y_true"], merged["ens_mu"], merged["ens_sigma"])
    loss_har = gaussian_nll_np(merged["y_true"], merged["har_mu"], merged["har_sigma"])
    loss_eg = gaussian_nll_np(merged["y_true"], merged["har_mu"], merged["egarch_sigma"])
    dm_stats = {
        "Ensemble vs HAR-style Gaussian": dm_test(loss_har, loss_ens),
        "Ensemble vs EGARCH": dm_test(loss_eg, loss_ens),
    }
    _json_dump(dm_stats, paths.reports / "dm_stats.json")

    quantile_row = pd.DataFrame([baseline_ctx["quantile_metrics"]], index=["Quantile-HAR"])
    quantile_row.to_csv(paths.tables / "quantile_metrics.csv")

    unc_year = (
        predictions.assign(year=pd.to_datetime(predictions["Date"]).dt.year)
        .groupby("year")[["aleatoric_sd", "epistemic_sd"]]
        .mean()
    )
    unc_year.to_csv(paths.tables / "uncertainty_by_year.csv")

    # Additional long-format metrics for plotting.
    density_long = density_table.reset_index().rename(columns={"index": "Model"})
    density_long.to_csv(paths.tables / "density_metrics_long.csv", index=False)

    # TeX fragments
    density_tex = density_table.round(3).to_latex(
        index=True,
        escape=False,
        bold_rows=False,
        column_format="lrrrrrrr",
    )
    (paths.tables / "density_metrics.tex").write_text(density_tex, encoding="utf-8")

    quantile_dm_lines = []
    quantile_dm_lines.append(r"\begin{tabular}{lrrrrrr}")
    quantile_dm_lines.append(r"\toprule")
    quantile_dm_lines.append(r"Model / test & Cov90 & IS90 & RMSE$_{0.50}$ & Pinball$_{0.05}$ & Pinball$_{0.50}$ & Pinball$_{0.95}$ \\")
    quantile_dm_lines.append(r"\midrule")
    q = baseline_ctx["quantile_metrics"]
    quantile_dm_lines.append(
        "Quantile-HAR & "
        f"{q['Cov90']:.3f} & {q['IS90']:.3f} & {q['RMSE_0.50']:.3f} & "
        f"{q['Pinball_0.05']:.3f} & {q['Pinball_0.50']:.3f} & {q['Pinball_0.95']:.3f} \\\\"
    )
    quantile_dm_lines.append(r"\bottomrule")
    quantile_dm_lines.append(r"\end{tabular}")
    quantile_dm_lines.append("")
    quantile_dm_lines.append(r"\vspace{0.8em}")
    quantile_dm_lines.append("")
    quantile_dm_lines.append(r"\begin{tabular}{lrr}")
    quantile_dm_lines.append(r"\toprule")
    quantile_dm_lines.append(r"NLL comparison & DM statistic & $p$-value \\")
    quantile_dm_lines.append(r"\midrule")
    quantile_dm_lines.append(
        f"Ensemble vs HAR-style Gaussian & {dm_stats['Ensemble vs HAR-style Gaussian']['stat']:.3f} & "
        f"{dm_stats['Ensemble vs HAR-style Gaussian']['p_value']:.3f} \\\\"
    )
    quantile_dm_lines.append(
        f"Ensemble vs EGARCH & {dm_stats['Ensemble vs EGARCH']['stat']:.3f} & "
        f"{dm_stats['Ensemble vs EGARCH']['p_value']:.3f} \\\\"
    )
    quantile_dm_lines.append(r"\bottomrule")
    quantile_dm_lines.append(r"\end{tabular}")
    (paths.tables / "quantile_and_dm.tex").write_text("\n".join(quantile_dm_lines), encoding="utf-8")

    unc_tex = unc_year.round(3).rename(
        columns={"aleatoric_sd": "Aleatoric s.d.", "epistemic_sd": "Epistemic s.d."}
    ).to_latex(index=True, column_format="lrr")
    (paths.tables / "uncertainty_by_year.tex").write_text(unc_tex, encoding="utf-8")

    summary = {
        "density_metrics": density_table.round(6).to_dict(orient="index"),
        "quantile_metrics": {k: float(v) for k, v in baseline_ctx["quantile_metrics"].items()},
        "dm_stats": dm_stats,
    }
    _json_dump(summary, paths.reports / "summary.json")

    return {
        "density_table": density_table,
        "quantile_table": quantile_row,
        "uncertainty_by_year": unc_year,
        "dm_stats": dm_stats,
        "all_predictions": merged,
    }


def _setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
        }
    )


def _save_figure(fig: plt.Figure, basepath: Path) -> None:
    fig.savefig(basepath.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(basepath.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_core_figures(
    paths: ProjectPaths,
    data_ctx: Dict[str, object],
    baseline_ctx: Dict[str, object],
    transformer_ctx: Dict[str, object],
    table_ctx: Dict[str, object],
    config: ExperimentConfig = CONFIG,
) -> None:
    _setup_plot_style()
    panel = data_ctx["panel"]
    y = data_ctx["target"]
    baseline_preds = baseline_ctx["baseline_predictions"]
    predictions = transformer_ctx["predictions"]
    merged = table_ctx["all_predictions"]

    # USD/MXN spot rate
    fig = plt.figure(figsize=(8, 4.6))
    plt.plot(panel.index, panel["Mexico"])
    plt.axvline(pd.Timestamp(config.train_end), linestyle="--", linewidth=1)
    plt.axvline(pd.Timestamp(config.val_end), linestyle="--", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("USD/MXN")
    plt.title("USD/MXN spot rate")
    plt.tight_layout()
    _save_figure(fig, paths.figures / "usdmxn_level")

    # Log volatility proxy
    fig = plt.figure(figsize=(8, 4.6))
    plt.plot(y.index, y.values)
    plt.axvline(pd.Timestamp(config.train_end), linestyle="--", linewidth=1)
    plt.axvline(pd.Timestamp(config.val_end), linestyle="--", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel(r"$\log(r_t^2 + \varepsilon)$")
    plt.title("Daily log-volatility proxy for USD/MXN")
    plt.tight_layout()
    _save_figure(fig, paths.figures / "logrv_proxy")

    # Learning curve
    hist_mean = transformer_ctx["ensemble_histories"].groupby("epoch")[["train_nll_std", "val_nll_std"]].mean().reset_index()
    fig = plt.figure(figsize=(8, 4.6))
    plt.plot(hist_mean["epoch"], hist_mean["train_nll_std"], label="Train")
    plt.plot(hist_mean["epoch"], hist_mean["val_nll_std"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Standardized NLL")
    plt.title("Transformer ensemble learning curve")
    plt.legend()
    plt.tight_layout()
    _save_figure(fig, paths.figures / "learning_curve")

    # Coverage calibration
    levels = np.arange(0.1, 1.0, 0.1)
    emp_cov_har = []
    emp_cov_ens = []
    for level in levels:
        alpha = 1 - level
        z = norm.ppf(1 - alpha / 2)
        lower_h = merged["har_mu"] - z * merged["har_sigma"]
        upper_h = merged["har_mu"] + z * merged["har_sigma"]
        lower_e = merged["ens_mu"] - z * merged["ens_sigma"]
        upper_e = merged["ens_mu"] + z * merged["ens_sigma"]
        emp_cov_har.append(np.mean((merged["y_true"] >= lower_h) & (merged["y_true"] <= upper_h)))
        emp_cov_ens.append(np.mean((merged["y_true"] >= lower_e) & (merged["y_true"] <= upper_e)))

    fig = plt.figure(figsize=(8, 4.6))
    plt.plot(levels, levels, linestyle="--", label="Ideal")
    plt.plot(levels, emp_cov_har, marker="o", label="HAR-style Gaussian")
    plt.plot(levels, emp_cov_ens, marker="o", label="Transformer ensemble")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title("Prediction interval calibration")
    plt.legend()
    plt.tight_layout()
    _save_figure(fig, paths.figures / "coverage_calibration")

    # Predictive mean vs realized
    fig = plt.figure(figsize=(8, 4.6))
    plt.plot(predictions["Date"], predictions["y_true"], label="Realized")
    plt.plot(predictions["Date"], predictions["ens_mu"], label="Predictive mean")
    plt.xlabel("Date")
    plt.ylabel(r"$\log(r_t^2 + \varepsilon)$")
    plt.title("Out-of-sample predictive mean versus realization")
    plt.legend()
    plt.tight_layout()
    _save_figure(fig, paths.figures / "pred_vs_realized")

    # PIT histogram
    pit = norm.cdf((predictions["y_true"] - predictions["ens_mu"]) / predictions["ens_sigma"])
    fig = plt.figure(figsize=(8, 4.6))
    plt.hist(pit, bins=10)
    plt.xlabel("PIT value")
    plt.ylabel("Frequency")
    plt.title("PIT histogram for transformer ensemble")
    plt.tight_layout()
    _save_figure(fig, paths.figures / "pit_hist")

    # Uncertainty decomposition
    fig = plt.figure(figsize=(8, 4.6))
    plt.plot(predictions["Date"], predictions["aleatoric_sd"], label="Aleatoric s.d.")
    plt.plot(predictions["Date"], predictions["epistemic_sd"], label="Epistemic s.d.")
    plt.xlabel("Date")
    plt.ylabel("Uncertainty on log-volatility scale")
    plt.title("Decomposition of predictive uncertainty")
    plt.legend()
    plt.tight_layout()
    _save_figure(fig, paths.figures / "uncertainty_decomp")

    # Zoomed uncertainty decomposition
    zoom = predictions[(predictions["Date"] >= "2020-01-01") & (predictions["Date"] <= "2021-01-31")]
    fig = plt.figure(figsize=(8, 4.6))
    plt.plot(zoom["Date"], zoom["aleatoric_sd"], label="Aleatoric s.d.")
    plt.plot(zoom["Date"], zoom["epistemic_sd"], label="Epistemic s.d.")
    plt.xlabel("Date")
    plt.ylabel("Uncertainty on log-volatility scale")
    plt.title("Uncertainty during the 2020 regime shift")
    plt.legend()
    plt.tight_layout()
    _save_figure(fig, paths.figures / "uncertainty_zoom_2020")

   


def plot_additional_figures(
    paths: ProjectPaths,
    data_ctx: Dict[str, object],
    table_ctx: Dict[str, object],
    config: ExperimentConfig = CONFIG,
) -> None:
    _setup_plot_style()
    log_rv = data_ctx["log_rv"]
    density_table = table_ctx["density_table"]

    # ACF of the target proxy.
    acf_vals = sm_acf(data_ctx["target"].values, nlags=30, fft=True)
    fig = plt.figure(figsize=(8, 4.6))
    lags = np.arange(len(acf_vals))
    plt.axhline(0.0, linewidth=1)
    plt.vlines(lags, 0.0, acf_vals)
    plt.plot(lags, acf_vals, marker="o")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation of the USD/MXN log-volatility proxy")
    plt.tight_layout()
    _save_figure(fig, paths.figures / "acf_vol_proxy")

    # Directed lagged spillover network from simple standardized regressions.
    train_log_rv = log_rv.loc[: pd.Timestamp(config.train_end)].copy()
    z = (train_log_rv - train_log_rv.mean()) / train_log_rv.std(ddof=0)
    lagged = z.shift(1).dropna()
    current = z.loc[lagged.index]
    coef_matrix = pd.DataFrame(index=config.countries, columns=config.countries, dtype=float)
    for target in config.countries:
        X = np.column_stack([np.ones(len(lagged)), lagged[list(config.countries)].values])
        y = current[target].values
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        coef_matrix.loc[target, list(config.countries)] = beta[1:]

    coef_matrix.to_csv(paths.tables / "lagged_spillover_matrix.csv")

    edges = []
    for target in config.countries:
        for source in config.countries:
            if source == target:
                continue
            weight = float(coef_matrix.loc[target, source])
            edges.append({"source": source, "target": target, "weight": weight, "abs_weight": abs(weight)})
    edge_df = pd.DataFrame(edges).sort_values("abs_weight", ascending=False)
    top_edges = edge_df.head(12)
    top_edges.to_csv(paths.tables / "lagged_spillover_top_edges.csv", index=False)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.axis("off")
    n = len(config.countries)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = {
        country: (np.cos(angle), np.sin(angle))
        for country, angle in zip(config.countries, angles)
    }

    max_abs = max(top_edges["abs_weight"].max(), 1e-6)
    for _, row in top_edges.iterrows():
        x1, y1 = coords[row["source"]]
        x2, y2 = coords[row["target"]]
        line_style = "-" if row["weight"] >= 0 else "--"
        bend = 0.15 if hash((row["source"], row["target"])) % 2 == 0 else -0.15
        ax.annotate(
            "",
            xy=(0.88 * x2, 0.88 * y2),
            xytext=(0.88 * x1, 0.88 * y1),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.0 + 4.5 * row["abs_weight"] / max_abs,
                linestyle=line_style,
                alpha=0.75,
                connectionstyle=f"arc3,rad={bend}",
                shrinkA=10,
                shrinkB=10,
            ),
        )

    for country, (x, y) in coords.items():
        circ = plt.Circle((x, y), 0.12, fill=False, linewidth=1.2)
        ax.add_patch(circ)
        ax.text(x, y, country.replace("South ", "S. "), ha="center", va="center", fontsize=10)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title("Lagged cross-currency volatility network\n(top standardized spillover coefficients)")
    ax.text(-1.28, -1.24, "Solid: positive spillover, dashed: negative spillover", fontsize=9)
    plt.tight_layout()
    _save_figure(fig, paths.figures / "fx_vol_network")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].axhline(0.0, linewidth=1)
    axes[0].vlines(lags, 0.0, acf_vals)
    axes[0].plot(lags, acf_vals, marker="o")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].set_title("ACF of the USD/MXN log-volatility proxy")

    net_ax = axes[1]
    net_ax.axis("off")
    for _, row in top_edges.iterrows():
        x1, y1 = coords[row["source"]]
        x2, y2 = coords[row["target"]]
        line_style = "-" if row["weight"] >= 0 else "--"
        bend = 0.15 if hash((row["source"], row["target"])) % 2 == 0 else -0.15
        net_ax.annotate(
            "",
            xy=(0.88 * x2, 0.88 * y2),
            xytext=(0.88 * x1, 0.88 * y1),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.0 + 4.5 * row["abs_weight"] / max_abs,
                linestyle=line_style,
                alpha=0.75,
                connectionstyle=f"arc3,rad={bend}",
                shrinkA=10,
                shrinkB=10,
            ),
        )
    for country, (x, y) in coords.items():
        circ = plt.Circle((x, y), 0.12, fill=False, linewidth=1.2)
        net_ax.add_patch(circ)
        net_ax.text(x, y, country.replace("South ", "S. "), ha="center", va="center", fontsize=9)
    net_ax.set_xlim(-1.35, 1.35)
    net_ax.set_ylim(-1.35, 1.35)
    net_ax.set_title("Lagged FX volatility network")
    net_ax.text(-1.25, -1.22, "Solid = positive, dashed = negative", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, paths.figures / "acf_network")

    # Metric comparison plots
    metric_specs = [
        ("NLL", "Negative log-likelihood", True, "metric_nll_comparison"),
        ("CRPS", "CRPS", True, "metric_crps_comparison"),
        ("QLIKE", "QLIKE", True, "metric_qlike_comparison"),
    ]
    for metric, ylabel, lower_better, filename in metric_specs:
        ordered = density_table[metric].sort_values(ascending=True if lower_better else False)
        fig = plt.figure(figsize=(8, 4.6))
        plt.bar(range(len(ordered)), ordered.values)
        plt.xticks(range(len(ordered)), ordered.index, rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"Model comparison: {ylabel}")
        plt.tight_layout()
        _save_figure(fig, paths.figures / filename)

    coverage_gap = (density_table["Cov90"] - 0.90).abs().sort_values()
    fig = plt.figure(figsize=(8, 4.6))
    plt.bar(range(len(coverage_gap)), coverage_gap.values)
    plt.xticks(range(len(coverage_gap)), coverage_gap.index, rotation=25, ha="right")
    plt.ylabel(r"$| \widehat{\mathrm{Cov}}_{0.90} - 0.90 |$")
    plt.title("Coverage-gap comparison at the 90% level")
    plt.tight_layout()
    _save_figure(fig, paths.figures / "metric_coverage_gap_comparison")

    # Combined score figure used in the paper.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    score_specs = [
        ("NLL", "NLL", True, axes[0, 0]),
        ("CRPS", "CRPS", True, axes[0, 1]),
        ("QLIKE", "QLIKE", True, axes[1, 0]),
    ]
    for metric, ylabel, lower_better, ax in score_specs:
        ordered = density_table[metric].sort_values(ascending=True if lower_better else False)
        ax.bar(range(len(ordered)), ordered.values)
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(ordered.index, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    ax = axes[1, 1]
    ax.bar(range(len(coverage_gap)), coverage_gap.values)
    ax.set_xticks(range(len(coverage_gap)))
    ax.set_xticklabels(coverage_gap.index, rotation=25, ha="right")
    ax.set_ylabel(r"$| \widehat{\mathrm{Cov}}_{0.90} - 0.90 |$")
    ax.set_title("Coverage gap")
    fig.tight_layout()
    _save_figure(fig, paths.figures / "score_comparison")


def sync_figures_to_paper(paths: ProjectPaths) -> None:
    core_names = [
        "acf_network",
        "coverage_calibration",
        "learning_curve",
        "logrv_proxy",
        "pit_hist",
        "pred_vs_realized",
        "score_comparison",
        "uncertainty_decomp",
        "uncertainty_zoom_2020",
        "usdmxn_level",
    ]
    for stem in core_names:
        src = paths.figures / f"{stem}.pdf"
        if src.exists():
            shutil.copy2(src, paths.paper_figures / src.name)


def compile_paper(paths: ProjectPaths) -> Dict[str, object]:
    sync_figures_to_paper(paths)
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    bibtex_ok = bibtex is not None and Path(bibtex).exists()

    result = {"compiled": False, "engine": None, "returncode": None}


    if latexmk and bibtex_ok:
        cmd = [latexmk, "-pdf", "-interaction=nonstopmode", "paper.tex"]
        proc = subprocess.run(cmd, cwd=paths.paper, capture_output=True, text=True)
        result = {"compiled": proc.returncode == 0, "engine": "latexmk", "returncode": proc.returncode}
        (paths.reports / "paper_build_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (paths.reports / "paper_build_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    elif pdflatex and bibtex_ok:
        stdout_parts = []
        stderr_parts = []
        ok = True
        for _ in range(2):
            proc = subprocess.run([pdflatex, "-interaction=nonstopmode", "paper.tex"], cwd=paths.paper, capture_output=True, text=True)
            stdout_parts.append(proc.stdout)
            stderr_parts.append(proc.stderr)
            ok = ok and proc.returncode == 0
        result = {"compiled": ok, "engine": "pdflatex", "returncode": 0 if ok else 1}
        (paths.reports / "paper_build_stdout.txt").write_text("\n\n".join(stdout_parts), encoding="utf-8")
        (paths.reports / "paper_build_stderr.txt").write_text("\n\n".join(stderr_parts), encoding="utf-8")
    else:
        reason = "No complete LaTeX toolchain found (latexmk/pdflatex+bibtex unavailable or unusable). Existing paper.pdf left unchanged.\n"
        (paths.reports / "paper_build_stdout.txt").write_text(reason, encoding="utf-8")
        (paths.reports / "paper_build_stderr.txt").write_text("", encoding="utf-8")

    _json_dump(result, paths.reports / "paper_build_result.json")
    return result


def write_manifest(paths: ProjectPaths) -> None:
    manifest_lines = [
        "USD/MXN volatility submission package",
        "",
        "Main entry points:",
        "  python run_all.py",
        "  python code/pipeline.py all",
        "",
        "Step-by-step scripts:",
        "  python code/01_prepare_data.py",
        "  python code/02_fit_baselines.py",
        "  python code/03_train_transformer.py",
        "  python code/04_build_tables.py",
        "  python code/05_make_figures.py",
        "  python code/06_build_paper.py",
        "",
        "Key outputs:",
        f"  processed data   -> {paths.processed.relative_to(paths.root)}",
        f"  model outputs    -> {paths.models.relative_to(paths.root)}",
        f"  predictions      -> {paths.predictions.relative_to(paths.root)}",
        f"  tables           -> {paths.tables.relative_to(paths.root)}",
        f"  figures          -> {paths.figures.relative_to(paths.root)}",
        f"  final paper      -> {paths.paper.relative_to(paths.root)}",
    ]
    (paths.root / "submission_manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")


def run_all(paths: Optional[ProjectPaths] = None, compile_after: bool = False) -> Dict[str, object]:
    paths = resolve_paths() if paths is None else paths
    data_ctx = prepare_data(paths)
    baseline_ctx = run_baselines(paths, data_ctx)
    transformer_ctx = train_transformers(paths, data_ctx, baseline_ctx)
    table_ctx = build_tables(paths, data_ctx, baseline_ctx, transformer_ctx)
    plot_core_figures(paths, data_ctx, baseline_ctx, transformer_ctx, table_ctx)
    plot_additional_figures(paths, data_ctx, table_ctx)
    build_result = compile_paper(paths) if compile_after else {"compiled": False, "engine": None}
    write_manifest(paths)
    return {
        "data": data_ctx["summary"],
        "baseline_metrics": baseline_ctx["density_metrics"],
        "transformer_metrics": {
            "univariate": transformer_ctx["uni_metrics"],
            "ensemble": transformer_ctx["ensemble_metrics"],
        },
        "paper_build": build_result,
    }


def cli() -> None:
    parser = argparse.ArgumentParser(description="USD/MXN volatility reproducibility pipeline")
    parser.add_argument(
        "stage",
        choices=["prepare", "baselines", "transformer", "tables", "figures", "paper", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--compile", action="store_true", help="Compile the paper when running stage 'paper' or 'all'.")
    args = parser.parse_args()

    paths = resolve_paths()

    if args.stage == "prepare":
        prepare_data(paths)
        write_manifest(paths)
        return

    data_ctx = prepare_data(paths)
    baseline_ctx = run_baselines(paths, data_ctx)

    if args.stage == "baselines":
        write_manifest(paths)
        return

    transformer_ctx = train_transformers(paths, data_ctx, baseline_ctx)

    if args.stage == "transformer":
        write_manifest(paths)
        return

    table_ctx = build_tables(paths, data_ctx, baseline_ctx, transformer_ctx)

    if args.stage == "tables":
        write_manifest(paths)
        return

    if args.stage == "figures":
        plot_core_figures(paths, data_ctx, baseline_ctx, transformer_ctx, table_ctx)
        plot_additional_figures(paths, data_ctx, table_ctx)
        write_manifest(paths)
        return

    if args.stage == "paper":
        plot_core_figures(paths, data_ctx, baseline_ctx, transformer_ctx, table_ctx)
        plot_additional_figures(paths, data_ctx, table_ctx)
        compile_paper(paths) if args.compile else sync_figures_to_paper(paths)
        write_manifest(paths)
        return

    if args.stage == "all":
        run_all(paths, compile_after=args.compile)
        return


if __name__ == "__main__":
    cli()
