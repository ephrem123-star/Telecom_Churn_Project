"""Train and evaluate the synthetic churn classifier reproducibly.
Run from project root: python scripts/train_churn_model.py
The test metrics are computed on a held-out 20% test split before the model is
refit on all synthetic rows for the sample dashboard and saved prediction app.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, MODEL_DIR, REPORT_DIR = ROOT / "data", ROOT / "models", ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
source = DATA_DIR / "synthetic_telecom_churn.csv"
if not source.exists():
    raise FileNotFoundError(f"Missing {source}. Run: python scripts/generate_churn_data.py")
df = pd.read_csv(source)

numeric_features = ["tenure_months", "monthly_spend", "data_usage_gb", "complaints_90d",
                    "network_issue_count", "usage_change_pct", "support_contacts_90d"]
categorical_features = ["contract_type", "payment_method", "region"]
features = numeric_features + categorical_features
X, y = df[features], df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
preprocess = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1)),
])
model.fit(X_train, y_train)
test_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.25  # demo choice only, not a recommended operational threshold
test_pred = (test_prob >= threshold).astype(int)
metrics = {
    "Accuracy at threshold 0.25": accuracy_score(y_test, test_pred),
    "Precision at threshold 0.25": precision_score(y_test, test_pred, zero_division=0),
    "Recall at threshold 0.25": recall_score(y_test, test_pred, zero_division=0),
    "F1 at threshold 0.25": f1_score(y_test, test_pred, zero_division=0),
    "ROC-AUC (threshold independent)": roc_auc_score(y_test, test_prob),
}
report = ["SYNTHETIC DATA ONLY — not real telecom model performance",
          "Test set: 1,000 records (20% hold-out); random_state=42.",
          "The 0.25 threshold is for demonstration, not a recommended business threshold.", ""]
report += [f"{name}: {value:.4f}" for name, value in metrics.items()]
report += ["", classification_report(y_test, test_pred, zero_division=0)]
(REPORT_DIR / "model_metrics.txt").write_text("\n".join(report), encoding="utf-8")

# Refit on all fictional rows for scoring the demo dashboard; metrics above remain held-out.
model.fit(X, y)
scores = model.predict_proba(X)[:, 1]
output = df.copy()
output["churn_probability"] = scores
output["risk_segment"] = pd.cut(scores, bins=[-0.001, 0.30, 0.60, 1.0],
    labels=["Low", "Medium", "High"], include_lowest=True).astype(str)
output.to_csv(DATA_DIR / "customer_churn_predictions.csv", index=False)
joblib.dump(model, MODEL_DIR / "churn_random_forest.joblib")
print("Saved model: models/churn_random_forest.joblib")
print("Saved scored rows: data/customer_churn_predictions.csv")
print("Saved evaluation: reports/model_metrics.txt")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
