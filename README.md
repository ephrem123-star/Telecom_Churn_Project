# Telecom Customer Churn Prediction & BI Demo

A portfolio demonstration using **5,000 synthetic (fictional) customer records**. It is not affiliated with Safaricom and uses no real operator customer data.

## Project structure

- `data/synthetic_telecom_churn.csv` — generated fictional source dataset
- `data/customer_churn_predictions.csv` — sample rows with model scores and risk segments
- `models/churn_random_forest.joblib` — fitted scikit-learn pipeline for the demo app
- `reports/model_metrics.txt` — held-out test metrics
- `scripts/generate_churn_data.py` — source-data generator
- `scripts/train_churn_model.py` — preprocessing, training, evaluation, and scoring source
- `app.py` — Streamlit single-customer demo

## Run locally (Python installed)

From the project root:

```bash
python -m pip install -r requirements.txt
python scripts/generate_churn_data.py
python scripts/train_churn_model.py
streamlit run app.py
```

The generator and training script use deterministic seeds. Training evaluates on a stratified 20% hold-out test set, then refits the pipeline on all synthetic rows for the dashboard/app demo. The reported test metrics are from before that final refit.

## Limitations

The labels are simulated from a formula, not observed real-world churn outcomes. Metrics measure how well a model learns those artificial labels and are not evidence of performance on real telecom customers. The 0.25 classification threshold is illustrative only. Do not use this demo for real customer decisions.
