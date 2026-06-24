# Company Performance Prediction — MLOps Pipeline

**Course:** MLOps and System Design (EADA, 2026)

---

## 1. Project Title

**Company Performance Prediction — A Reproducible Batch MLOps Pipeline**

We built an end-to-end machine learning pipeline that forecasts each company's **next-year revenue growth** from its current-year fundamentals, stock behaviour, and macroeconomic context. The system is designed as a **batch, offline-first** workflow with MLflow experiment tracking, automated testing, and continuous deployment of retrained model artifacts.

---

## 2. Problem Statement

### 2.1 Business problem

Investors and analysts routinely ask: *given what we know about a company today, how will it perform next year?* We frame this as a **supervised regression** problem on a **company–year panel**: each row is one company at fiscal year **t**, and the label is performance in year **t+1**.

The course grades **pipeline mechanics** (structure, reproducibility, MLflow, CI/CD, documentation) rather than forecasting accuracy. Our goal is a runnable, auditable system a professor can execute on any machine without live API calls.

### 2.2 Target variable

| Decision | Choice | Rationale |
|---|---|---|
| Primary target | `target_revenue_growth = revenue_{t+1} / revenue_t − 1` | Revenue growth is interpretable, comparable across sectors, and aligned with the course brief |
| Alternatives considered | Net-income growth, operating margin | Supported in `config.yaml` but not used for the final model |
| Label timing | Year **t+1** only, via `shift(-1)` per company | Prevents leakage from future information |

### 2.3 Data strategy

| Decision | Choice | Rationale |
|---|---|---|
| Panel size | 500 companies × fiscal years 2018–2026 | Large enough for batch ML; small enough for fast CI |
| Default data mode | Committed **synthetic snapshots** (`fetch.use_live_apis: false`) | Reproducible runs with no network dependency in tests or CI |
| Optional refresh | yfinance + FRED (`pandas-datareader`) | Demonstrates live ingestion without making it mandatory |
| Raw artifacts | `fundamentals_snapshot.csv`, `stock_snapshot.csv`, `macro_snapshot.csv`, unified panel | Clear separation between ingestion and feature engineering |

### 2.4 Feature engineering

| Decision | Choice | Rationale |
|---|---|---|
| Feature year | Year **t** data only | Strict leakage control |
| Engineered signals | 18 features: profitability ratios, returns, leverage, market metrics, macro indicators, encoded ticker | Compress raw financials into modelling-ready signals |
| Exclusions | IDs, raw line items, all `target_*` columns | Targets must never enter `feature_columns()` |
| Output | `datasets/processed/features.csv` (~4,000 rows) | 500 companies × 8 labelled years (2026 raw year has no t+1 label) |

### 2.5 Train / test design

| Decision | Choice | Rationale |
|---|---|---|
| Split type | **Chronological** — no shuffle | Mimics real forecasting: train on the past, evaluate on a future holdout |
| Training years | 2018–2022 (2,500 rows) | Five years of history per company |
| Test holdout | Fiscal year **2023** (500 rows) | Labels reflect 2024 performance; simulates backtesting |
| Forward scoring (production) | Feature year **2025** → predict **2026** | Separate on-demand batch workflow, not the same as the 2023 evaluation holdout |

### 2.6 MLOps and system design

| Decision | Choice | Rationale |
|---|---|---|
| CLI entrypoint | `main.py` with `fetch`, `features`, `train`, `predict`, `all` | Single interface for the full pipeline |
| Experiment tracking | MLflow (`sqlite:///mlflow.db`, experiment `company-performance-prediction`) | Compare runs and satisfy course MLflow requirement |
| Model persistence | `models/model.joblib` + `models/metadata.json` | Versioned artifacts for predict and CD |
| CI | flake8 + offline `pytest` on pull requests | Catch regressions without network calls |
| CD | `features` + `train` on push to `main`, commit updated `models/` | Automated retraining in deployment flow |
| On-demand predictions | `batch_prediction_dataset/on_demand_predictions.csv` | Course requirement for batch scoring |

### 2.7 Pipeline steps

```
fetch  →  features  →  train  →  predict
```

1. **`fetch`** — Load or generate raw company-year snapshots into `datasets/raw/`.
2. **`features`** — Engineer leakage-safe features; write `features.csv` and the on-demand scoring set.
3. **`train`** — Compare four sklearn models with MLflow; persist the best model by test RMSE.
4. **`predict`** — Score the 2025 feature year and write 2026 revenue-growth predictions for 500 companies.

---

## 3. Model Development

### 3.1 Models compared

All models use the same preprocessing pipeline: **`StandardScaler` + estimator**. Four algorithms were trained and logged to MLflow:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Evaluation metrics:** MAE, RMSE, R², and directional accuracy (fraction of predictions with the correct sign vs. the actual target).

### 3.2 Test-set results (holdout year 2023)

| Model | MAE | RMSE | R² | Directional accuracy |
|---|---|---|---|---|
| Linear Regression | 0.0906 | 0.1125 | −0.0113 | 0.688 |
| **Ridge Regression** | **0.0906** | **0.1125** | **−0.0112** | **0.688** |
| Random Forest | 0.0932 | 0.1162 | −0.0790 | 0.668 |
| Gradient Boosting | 0.0915 | 0.1143 | −0.0437 | 0.684 |

*Source: `models/metadata.json` — test partition, fiscal year 2023 (labels = 2024 performance).*

### 3.3 Why Ridge was selected

**Ridge regression** was chosen as the production model because it achieved the **lowest test RMSE (0.1125)** among all candidates.

Tree-based models (Random Forest, Gradient Boosting) showed clear **overfitting**: strong train R² (up to 0.85 for Random Forest) but worse test RMSE and negative test R². Ridge and Linear Regression generalised best on the chronological holdout; Ridge edged out Linear Regression on RMSE while applying mild L2 regularisation, which is preferable when features are correlated (financial ratios and macro variables).

Model selection criterion: **lowest test RMSE** on the 2023 holdout (configured via `training.test_year` in `config.yaml`).

### 3.4 MLflow experiments

Runs are logged to the `company-performance-prediction` experiment. To reproduce the UI locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 and select the experiment to inspect parameters and metrics per run.

**Experiment runs (latest test metrics per model):**

![MLflow experiment runs](docs/images/mlflow-experiment-runs.png)

**Metric comparison across models:**

![MLflow metrics comparison](docs/images/mlflow-metrics-comparison.png)

Additional analysis is available in `notebooks/model_experiments.ipynb`.

---

## 4. Conclusions

We delivered a **reproducible batch MLOps pipeline** that ingests company-year data, engineers leakage-safe features, trains and compares four regression models with MLflow tracking, and produces on-demand predictions for fiscal year 2026.

**Key components:**

- Offline-first data layer with committed snapshots and a configurable live-fetch path
- Feature table of ~4,000 labelled rows and a separate 500-row forward-scoring batch (feature year 2025)
- Chronological evaluation on fiscal year 2023, distinct from production-style 2026 scoring
- MLflow experiment logging, CI (lint + pytest), and CD (retrain + commit `models/`)
- Persisted artifacts: `models/model.joblib`, `models/metadata.json`, `batch_prediction_dataset/on_demand_predictions.csv`

**Final model:** Ridge regression — test RMSE **0.1125**, MAE **0.0906**, directional accuracy **68.8%** on the 2023 holdout.

**Forward predictions:** The pipeline scored **500 companies** for **2026** revenue growth (using 2025 features). Mean predicted growth ≈ **5.0%** (range ≈ 2.3%–8.6%). These are batch outputs for demonstration; the course emphasises pipeline design over alpha generation.

Run the full pipeline:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py all
pytest tests/ -v
```
