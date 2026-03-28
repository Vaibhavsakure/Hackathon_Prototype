import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

df = pd.read_csv("data/master_df.csv")
decision_vars = ["Granulation_Time", "Binder_Amount", "Drying_Temp", "Drying_Time", "Compression_Force", "Machine_Speed", "Lubricant_Conc"]
outcomes = ["Quality_Score", "Yield_Score", "Total_Energy_kWh", "Carbon_kg_CO2", "Performance_Score"]

X = df[decision_vars].values
results = {"targets": {}}

for outcome in outcomes:
    y = df[outcome].values
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=6)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_preds = cross_val_predict(model, X, y, cv=5)
    
    model.fit(X, y)
    full_preds = model.predict(X)
    
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    
    results["targets"][outcome] = {
        "r2_cv": float(cv_scores.mean()),
        "r2_full": float(r2_score(y, full_preds)),
        "mae": float(mean_absolute_error(y, cv_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, cv_preds))),
        "top_features": {decision_vars[i]: float(importances[i]) for i in idx[:3]}
    }

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
