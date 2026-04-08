"""
Train at-risk student detection model.
Run once: python scripts/train_risk_model.py
Saves model to models/risk_model.pkl
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

np.random.seed(42)
N = 300

# Simulate realistic student data
grade           = np.clip(np.random.normal(70, 18, N), 0, 100)
attendance      = np.clip(np.random.normal(78, 15, N), 0, 100)
assignments     = np.random.randint(0, 9, N)
quiz_avg        = np.clip(grade + np.random.normal(0, 10, N), 0, 100)

# Label: at-risk if any major red flag
at_risk = (
    (grade < 60) |
    (attendance < 65) |
    (assignments < 3) |
    ((grade < 70) & (attendance < 75) & (assignments < 5))
).astype(int)

df = pd.DataFrame({
    "grade":        grade,
    "attendance":   attendance,
    "assignments":  assignments,
    "quiz_avg":     quiz_avg,
    "at_risk":      at_risk,
})

print(f"Dataset: {N} students, {at_risk.sum()} at-risk ({at_risk.mean()*100:.1f}%)")

X = df[["grade", "attendance", "assignments", "quiz_avg"]]
y = df["at_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(class_weight="balanced", random_state=42)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not At Risk", "At Risk"]))

# Feature importance (coefficients after scaling)
coefs = pipeline.named_steps["clf"].coef_[0]
features = ["grade", "attendance", "assignments", "quiz_avg"]
print("\nFeature importance:")
for f, c in sorted(zip(features, coefs), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {f}: {c:.3f}")

os.makedirs("models", exist_ok=True)
with open("models/risk_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\n Model saved to models/risk_model.pkl")