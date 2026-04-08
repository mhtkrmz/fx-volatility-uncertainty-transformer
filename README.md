# Distributional Forecasting of FX Volatility with Aleatoric and Epistemic Uncertainty

A clean GitHub-ready repository scaffold for the paper **"Distributional Forecasting of FX Volatility with Aleatoric and Epistemic Uncertainty"**.


Probabilistic forecasting of USD/MXN volatility with a patch-based Transformer, deep ensembles, and explicit aleatoric/epistemic uncertainty decomposition.

## Overview

This project studies **next-day USD/MXN volatility forecasting as a distributional prediction problem** rather than a point-estimation task. The paper proposes a **compact probabilistic patch-based Transformer** that maps the previous **40 trading days** of multivariate FX data into a **Gaussian forecast for next-day log-volatility**. A **deep ensemble** is then used to separate **aleatoric uncertainty** (market noise) from **epistemic uncertainty** (model uncertainty).

The empirical setup compares the Transformer against strong classical baselines:

- HAR-style Gaussian
- GARCH(1,1)
- GJR-GARCH
- EGARCH
- Quantile-HAR

## Main takeaway

The paper's core result is intentionally balanced:

- Classical volatility models remain strongest overall on the daily squared-return proxy.
- The multivariate Transformer improves over a univariate Transformer.
- The Transformer still produces useful **calibrated intervals** and informative **epistemic-uncertainty spikes** during regime shifts.

This makes the framework especially interesting as a **risk-management overlay** even when it does not win the forecasting horse race.

## Paper summary

### Forecast target

The target is next-day log-volatility for USD/MXN, defined from the daily squared-return proxy:

```text
y_t = log(r_t^2 + ε)
```

with:

- daily returns from USD/MXN spot data
- `ε = 1e-8`
- one-step-ahead conditional density as the forecasting object

### Inputs

Each day uses a **23-dimensional feature vector** including:

- cross-currency returns
- cross-currency log-volatility proxies
- day-of-week dummies
- HAR-style weekly and monthly aggregates

Currencies included alongside USD/MXN:

- BRL
- CAD
- EUR
- ZAR
- JPY
- KRW
- CNY

### Transformer design

The proposed model uses:

- lookback window: **40 trading days**
- patch size: **5 days**
- number of patches: **8**
- encoder layers: **1**
- attention heads: **4**
- embedding dimension: **16**
- feed-forward width: **32**
- Gaussian output head: predicts `μ_t` and `σ_t^2`
- ensemble size: **3**

### Data split

Chronological split:

- **Train:** through 2015-12-31
- **Validation:** 2016-01-01 to 2019-12-31
- **Test:** 2020-01-01 onward

### Key findings

- **EGARCH** achieves the best test NLL.
- **HAR-style Gaussian** is strongest on CRPS, QLIKE, and interval score among the parametric density models.
- The **Transformer ensemble** improves over the univariate Transformer but still trails the classical baselines overall.
- The Transformer's uncertainty decomposition remains useful for identifying model disagreement during regime shifts.


## Installation

```bash
git clone <your-repo-url>
cd fx-volatility-uncertainty-transformer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Minimal requirements

This scaffold includes a starter `requirements.txt`. You will likely want packages such as:

- numpy
- pandas
- scipy
- scikit-learn
- statsmodels
- arch
- matplotlib
- seaborn
- torch
- tqdm
- pyyaml

## GitHub topics

```text
fx-volatility
volatility-forecasting
transformer
uncertainty-quantification
deep-ensembles
garch
har-model
risk-management
time-series
probabilistic-forecasting
```

