# RUC-TS

**Randomized Uphill Climbing for Automated Feature Program Synthesis in Time-Series Forecasting**

## Overview

RUC-TS treats feature engineering as **stochastic program synthesis** over a domain-specific grammar. Starting from 370+ operators, it searches for high-value symbolic feature expressions through Randomized Uphill Climbing -- a local-search strategy pairing Metropolis acceptance with simulated annealing.

**Key results:** 13.0-19.0% MAE reduction over raw features, 5.2x faster than GP, fully interpretable expressions.

## Architecture

```
Phase 1: Initialization          Phase 2: RUC Search              Phase 3: Scoring & Filter
                                  (I=200 iterations)
Multivariate Time Series    -->  Elite Selection (Top-50)    -->  OLS Surrogate (Fast)
Operator Grammar (370+ ops)      Perturbation (4 mutations)       MLP Surrogate (Nonlinear)
Depth <= 4                       Metropolis Acceptance             s = 0.6*s_OLS + 0.4*s_MLP
                                 Diversity Injection (10%)         VIF < 5 / Stability Filter
                                 Annealing: tau = tau0*(1-t/I)    --> K=30 symbolic features
```

## Installation

```bash
pip install -e ".[dev]"
```

## Experiments

```bash
# Table 2: Main results (3 datasets x 3 models)
python experiments/run_main.py --seed 42

# Figure 5a: Ablation study
python experiments/run_ablation.py --seed 42

# Figure 4: Convergence analysis
python experiments/run_convergence.py --seed 42
```

## Main Results (Table 2)

| Features | S&P/XGB | S&P/LSTM | S&P/TFT | Ener/XGB | Ener/LSTM | Ener/TFT | Jena/XGB | Jena/LSTM | Jena/TFT |
|----------|---------|----------|---------|----------|-----------|----------|----------|-----------|----------|
| Raw      | .0341   | .0358    | .0332   | 68.42    | 65.18     | 61.73    | 2.184    | 1.973     | 1.842    |
| GP       | .0298   | .0314    | .0291   | 59.87    | 57.12     | 54.38    | 1.923    | 1.782     | 1.689    |
| **RUC-TS** | **.0278** | **.0292** | **.0270** | **55.41** | **53.18** | **50.87** | **1.824** | **1.697** | **1.603** |

## Tests

```bash
pytest tests/ -v
```

## License

Apache 2.0
