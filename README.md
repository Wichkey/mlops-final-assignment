# Company Performance Prediction

**Course:** MLOps and System Design (EADA, 2026)

Batch ML pipeline that predicts next-year company revenue growth from fundamentals, stock data, and macro indicators.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py all
pytest tests/ -v
```

| Command | Description |
|---|---|
| `python main.py fetch` | Load or generate raw CSV snapshots |
| `python main.py features` | Engineer features |
| `python main.py train` | Train models and log to MLflow |
| `python main.py predict` | On-demand batch predictions |
| `python main.py all` | Run the full pipeline |

## Documentation

Full project documentation (problem statement, version control, model development, CI/CD, on-demand workflow):

**[documentation.md](documentation.md)**

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
