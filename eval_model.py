"""Evaluate RandomForestRegressor accuracy on AuraOptima data."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

# Load data
df = pd.read_csv("data/master_df.csv")

decision_vars = [
    "Granulation_Time", "Binder_Amount", "Drying_Temp",
    "Drying_Time", "Compression_Force", "Machine_Speed", "Lubricant_Conc"
]
outcomes = ["Quality_Score", "Yield_Score", "Total_Energy_kWh", "Carbon_kg_CO2", "Performance_Score"]

X = df[decision_vars].values
n_samples = X.shape[0]

print("=" * 65)
print("  AuraOptima - RandomForestRegressor Evaluation Report")
print("=" * 65)
print(f"\nDataset: {n_samples} batches, {len(decision_vars)} features, {len(outcomes)} targets")
print(f"Model: RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)")
print(f"Evaluation: 5-fold Cross Validation + Full-fit metrics\n")
print("-" * 65)
print(f"{'Target':<22} {'R2 (CV)':<12} {'R2 (Full)':<12} {'MAE':<10} {'RMSE':<10}")
print("-" * 65)

all_r2_cv = []
all_r2_full = []

for outcome in outcomes:
    y = df[outcome].values
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=6)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    r2_cv = cv_scores.mean()
    
    cv_preds = cross_val_predict(model, X, y, cv=5)
    mae = mean_absolute_error(y, cv_preds)
    rmse = np.sqrt(mean_squared_error(y, cv_preds))
    
    model.fit(X, y)
    full_preds = model.predict(X)
    r2_full = r2_score(y, full_preds)
    
    all_r2_cv.append(r2_cv)
    all_r2_full.append(r2_full)
    
    print(f"{outcome:<22} {r2_cv:<12.4f} {r2_full:<12.4f} {mae:<10.3f} {rmse:<10.3f}")

print("-" * 65)
print(f"{'AVERAGE':<22} {np.mean(all_r2_cv):<12.4f} {np.mean(all_r2_full):<12.4f}")
print()

print("=" * 65)
print("  Feature Importance (per target)")
print("=" * 65)

for outcome in outcomes:
    y = df[outcome].values
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=6).fit(X, y)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print(f"\n  {outcome}:")
    for idx in sorted_idx:
        bar = "#" * int(importances[idx] * 40)
        print(f"   {decision_vars[idx]:<22} {importances[idx]:.3f}  {bar}")

print("\n" + "=" * 65)
print("  Model Config")
print("=" * 65)
print(f"  n_estimators:   50")
print(f"  max_depth:      6")
print(f"  n_features:     7")
print(f"  criterion:      squared_error")
print(f"  random_state:   42")
print()
