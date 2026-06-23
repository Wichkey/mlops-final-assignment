# Company Performance Prediction — MLOps Pipeline

**Course:** MLOps and System Design (EADA, 2026)  
**Team:** _[Teammate 1]_, _[Teammate 2]_, _[Teammate 3]_

> **For all contributors:** this README is the single source of truth for what we are building and how it should work. Read it before every change. When using Cursor, start your prompt with: *"Read README.md and implement only the scope described below."*

---

## 1. What this project is

We build a **reproducible batch ML pipeline** that predicts a company's **next-year performance** from its current-year fundamentals, stock behaviour, and macroeconomic context.

- **Problem type:** regression on a cross-sectional company-year panel
- **Default target:** next-year revenue growth  
  `target_revenue_growth = revenue_{t+1} / revenue_t - 1`
- **Grading focus:** pipeline mechanics (structure, MLflow, CI/CD, reproducibility, documentation) — not forecasting accuracy

Each row is one **(company, fiscal_year)** observation at year **t**. The label always comes from year **t+1**.

---

## 2. How the pipeline should work

End-to-end flow:

```
fetch  →  features  →  train  →  predict
  │           │           │          │
  ▼           ▼           ▼          ▼
raw/     processed/    models/   batch_prediction_dataset/
CSV      features.csv  joblib    on_demand_predictions.csv
```

### Stage 1 — `python main.py fetch`

**Owner:** Teammate 1 (data)

- Pull or generate a company-year panel and write CSV snapshots to `datasets/raw/`:
  - `fundamentals_snapshot.csv` — income statement, balance sheet, cash flow line items
  - `stock_snapshot.csv` — annual return, volatility, market cap
  - `macro_snapshot.csv` — GDP growth, 10Y yield, CPI, unemployment (by fiscal year)
  - `synthetic_company_year_panel.csv` — unified offline panel
- **Default:** `fetch.use_live_apis: false` — use committed synthetic snapshots (no network in CI)
- **Optional live refresh:** yfinance (~150 global large caps) + FRED via `pandas-datareader`
- Synthetic panel: **500 companies**, fiscal years **2018–2026**

### Stage 2 — `python main.py features`

**Owner:** Teammate 1 (data)

- Read raw snapshots, engineer leakage-safe features, write `datasets/processed/features.csv`
- Build `batch_prediction_dataset/on_demand_dataset.csv` from `prediction.feature_year`
- **Leakage rules (mandatory):**
  - Features use only year **t** data
  - Targets use year **t+1** via `shift(-1)` inside each company `groupby`
  - `feature_columns()` must exclude: ids, raw financial line items, all `target_*` columns

**Engineered features (year t):**

| Category | Columns |
|---|---|
| Profitability | gross_margin, operating_margin, net_margin |
| Returns | roa, roe |
| Leverage / liquidity | debt_to_equity, current_ratio |
| Cash / efficiency | fcf_margin, asset_turnover |
| Dynamics | prior_revenue_growth |
| Market | annual_return, annual_volatility, log_market_cap |
| Macro | gdp_growth, dgs10, cpi, unrate |
| Categorical | ticker_code |

**Expected output size:** ~4,000 rows (500 companies × 8 feature years; latest raw year dropped because it has no t+1 label).

### Stage 3 — `python main.py train`

**Owner:** Teammate 2 (modelling)

- Chronological train/test split — **no shuffle**
  - **Train:** fiscal years 2018–2022 (2,500 rows)
  - **Test holdout:** fiscal year **2023** (500 rows); labels are 2024 performance
  - Configured via `training.test_year: 2023`
- Compare four sklearn models via MLflow:
  - linear_regression, ridge, random_forest, gradient_boosting
- Pipeline: `StandardScaler` + estimator
- Metrics: **MAE, RMSE, R², directional_accuracy**
- MLflow: `sqlite:///mlflow.db`, experiment `company-performance-prediction`
- Save best model (lowest RMSE) to `models/model.joblib` + `models/metadata.json`

### Stage 4 — `python main.py predict`

**Owner:** Teammate 3 (MLOps)

- **On-demand workflow (course requirement):**
  - Input: `batch_prediction_dataset/on_demand_dataset.csv`
  - Output: `batch_prediction_dataset/on_demand_predictions.csv`
- Score the latest feature year in the panel:
  - `prediction.feature_year: 2025` → predict `prediction.target_year: 2026`
- This is separate from the 2023 test holdout (production-style forward scoring).

### Stage 5 — `python main.py all`

Runs fetch → features → train → predict in sequence.

---

## 3. Configuration (`config.yaml`)

All paths, tickers, and hyperparameters live in `config.yaml` (created by Teammate 1, commit 2).

Key settings:

```yaml
target:
  type: revenue_growth          # or net_income_growth, operating_margin

training:
  test_year: 2023

fetch:
  use_live_apis: false
  n_companies: 500
  synthetic_years: [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

prediction:
  feature_year: 2025
  target_year: 2026
```

Shared helpers go in `src/utils.py`: `load_config()`, `ensure_dir()`, `setup_logging()`, `get_target_column()`.

---

## 4. Target repository layout

```
main.py                          # CLI: fetch | features | train | predict | all
config.yaml
requirements.txt                 # pinned dependencies
src/
  utils.py
  data/fetch.py
  features/build_features.py
  models/train.py
  models/predict.py
datasets/raw/                    # committed CSV snapshots
datasets/processed/              # features.csv
models/                          # model.joblib, metadata.json
batch_prediction_dataset/        # on_demand_dataset.csv, on_demand_predictions.csv
tests/test_pipeline.py
notebooks/model_experiments.ipynb  # Jupyter + MLflow (course requirement)
.github/workflows/
  ci.yml                         # lint + pytest on pull_request
  cd.yml                         # features + train on push to main
PROJECT_MEMORY.md                # course deliverable markdown (Teammate 3)
```

---

## 5. CI/CD (course requirement)

| Workflow | Trigger | Must do |
|---|---|---|
| **CI** | `pull_request` → main | `pip install`, flake8, offline `pytest` |
| **CD** | `push` → main | `python main.py features && python main.py train`, commit updated `models/` |

CI must never depend on live API calls.

---

## 6. Tests (offline, no network)

**Owner:** Teammate 3

`pytest tests/ -v` must assert:

- Feature table is non-empty (~4,000 rows)
- No NaNs in feature columns
- Target columns are not in `feature_columns()`
- Train/test split is chronological; test year == 2023
- On-demand dataset uses `prediction.feature_year`
- Training logs all four metrics; `metadata.json` lists four models

---

## 7. Course deliverables checklist

| Deliverable | Where |
|---|---|
| Problem statement & system design | `PROJECT_MEMORY.md` |
| Model comparison + MLflow screenshots | `PROJECT_MEMORY.md` + `notebooks/model_experiments.ipynb` |
| `requirements.txt` with versions | repo root |
| CI/CD pipelines | `.github/workflows/` |
| On-demand predictions | `batch_prediction_dataset/` |
| Runnable on professor's machine | `python main.py all` + committed snapshots |

---

## 8. Implementation ownership (commit plan)

| Teammate | Commits | Scope |
|---|---|---|
| **1** | 4 | Scaffold + README → config/utils → fetch/snapshots → features |
| **2** | 3 | train + split → MLflow/model artifacts → experiment notebook |
| **3** | 3 | tests → on-demand predict → CI/CD + PROJECT_MEMORY.md |

---

## 9. Quick start (once pipeline is implemented)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py all
pytest tests/ -v
```

**MLflow UI:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 10. Design constraints

- **Batch, not real-time:** live fetch may take minutes; CI uses committed CSVs.
- **Reproducibility:** synthetic panel ships with the repo; `random_seed: 42`.
- **No leakage:** strict feature/target separation; chronological test year 2023.
- **Honest scope:** yfinance history is shallow and restated — fine for a course demo, not production alpha.
