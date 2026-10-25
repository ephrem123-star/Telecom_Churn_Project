"""Generate 5,000 fictional telecom customer records for a portfolio demo.
All rows and churn labels are synthetic; they are not from Safaricom or any operator.
Run from any directory: python scripts/generate_churn_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
n = 5000

df = pd.DataFrame({
    "customer_id": [f"CUST{i:05d}" for i in range(1, n + 1)],
    "tenure_months": rng.integers(1, 73, size=n),
    "monthly_spend": np.clip(rng.normal(650, 220, size=n), 100, 1600).round(2),
    "data_usage_gb": np.clip(rng.gamma(2.2, 3.2, size=n), 0.1, 35).round(2),
    "complaints_90d": np.clip(rng.poisson(0.45, size=n), 0, 5),
    "network_issue_count": np.clip(rng.poisson(0.8, size=n), 0, 7),
    "usage_change_pct": np.clip(rng.normal(-2, 18, size=n), -70, 70).round(2),
    "contract_type": rng.choice(["Prepaid", "Monthly", "Annual"], size=n, p=[0.48, 0.34, 0.18]),
    "payment_method": rng.choice(["Mobile Money", "Bank", "Cash/Agent", "Card"], size=n, p=[0.46, 0.18, 0.28, 0.08]),
    "region": rng.choice(["North", "South", "East", "West", "Central"], size=n),
    "support_contacts_90d": np.clip(rng.poisson(0.7, size=n), 0, 6),
})

logit = (
    -2.15 + 0.035 * df["complaints_90d"] + 0.16 * df["network_issue_count"]
    + 0.018 * np.maximum(-df["usage_change_pct"], 0)
    - 0.028 * np.minimum(df["tenure_months"], 36)
    + 0.55 * (df["contract_type"] == "Prepaid")
    + 0.45 * (df["contract_type"] == "Monthly")
    + 0.10 * (df["monthly_spend"] > 900)
    + 0.08 * df["support_contacts_90d"]
)
p = 1 / (1 + np.exp(-logit))
df["churn"] = rng.binomial(1, p)
df["churn_status"] = np.where(df["churn"] == 1, "Churned", "Active")
df["estimated_monthly_revenue"] = df["monthly_spend"]
out = DATA_DIR / "synthetic_telecom_churn.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df):,} fictional records to {out}")
print(f"Synthetic churn rate: {df['churn'].mean():.1%}")
