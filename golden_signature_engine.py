import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler

# ─── CONFIG ───────────────────────────────────────────────
MASTER_CSV      = "data/master_df.csv"
SIGNATURES_FILE = "data/golden_signatures.json"

DECISION_VARS = [
    "Granulation_Time", "Binder_Amount", "Drying_Temp",
    "Drying_Time", "Compression_Force", "Machine_Speed", "Lubricant_Conc"
]

df = pd.read_csv(MASTER_CSV)

# ─── NORMALIZE ────────────────────────────────────────────
scaler = MinMaxScaler()
df["norm_quality"] = scaler.fit_transform(df[["Quality_Score"]])
df["norm_yield"]   = scaler.fit_transform(df[["Yield_Score"]])
df["norm_energy"]  = 1 - scaler.fit_transform(df[["Total_Energy_kWh"]])
df["norm_carbon"]  = 1 - scaler.fit_transform(df[["Carbon_kg_CO2"]])

# ─── MODE DEFINITIONS ─────────────────────────────────────
MODES = {
    "energy": {
        "label":        "Best Yield + Lowest Energy",
        "weights":      {"quality": 0.1, "yield": 0.2, "energy": 0.7},
        "primary_col":  "Total_Energy_kWh",   # minimize
        "primary_dir":  "min"
    },
    "quality": {
        "label":        "Optimal Quality + Best Yield",
        "weights":      {"quality": 0.7, "yield": 0.2, "energy": 0.1},
        "primary_col":  "Quality_Score",       # maximize
        "primary_dir":  "max"
    },
    "balanced": {
        "label":        "Max Performance + Min Carbon",
        "weights":      {"quality": 0.35, "yield": 0.35, "energy": 0.3},
        "primary_col":  "Performance_Score",   # maximize
        "primary_dir":  "max"
    }
}

# ─── PARETO FRONT FROM REAL DATA ──────────────────────────
def get_pareto_front_indices(objectives_df):
    """
    objectives_df columns must all be in MINIMIZATION form.
    Returns list of row indices that are Pareto-optimal.
    """
    values = objectives_df.values
    n = len(values)
    pareto = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is <= in all and < in at least one
            if all(values[j] <= values[i]) and any(values[j] < values[i]):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto

# ─── BUILD SIGNATURE FOR ONE MODE ─────────────────────────
def build_signature(mode_key):
    cfg     = MODES[mode_key]
    weights = cfg["weights"]

    # ── Step 1: Find best batch for THIS mode's primary objective ──
    if cfg["primary_dir"] == "min":
        best_idx = df[cfg["primary_col"]].idxmin()
    else:
        best_idx = df[cfg["primary_col"]].idxmax()

    best_batch = df.loc[best_idx]

    # ── Step 2: Composite score of that best batch ──
    composite = (
        weights["quality"] * best_batch["norm_quality"] +
        weights["yield"]   * best_batch["norm_yield"]   +
        weights["energy"]  * best_batch["norm_energy"]
    )

    # ── Step 3: Build Pareto front (minimize neg_quality, neg_yield, energy) ──
    pareto_input = pd.DataFrame({
        "neg_quality": -df["norm_quality"],
        "neg_yield":   -df["norm_yield"],
        "energy":      -df["norm_energy"]   # norm_energy already inverted, so negate again
    })
    pareto_idx    = get_pareto_front_indices(pareto_input)
    pareto_df     = df.iloc[pareto_idx].copy()

    pareto_points = []
    for _, row in pareto_df.iterrows():
        pareto_points.append({
            "batch_id": row["Batch_ID"],
            "quality":  round(row["Quality_Score"], 2),
            "yield":    round(row["Yield_Score"], 2),
            "energy":   round(row["Total_Energy_kWh"], 2),
            "carbon":   round(row["Carbon_kg_CO2"], 2),
        })

    # ── Step 4: Assemble signature ──
    signature = {
        "mode":    mode_key,
        "label":   cfg["label"],
        "version": 1,
        "status":  "active",
        "weights": weights,

        "optimal_params": {
            var: round(float(best_batch[var]), 3)
            for var in DECISION_VARS
        },

        "benchmark_scores": {
            "Quality_Score":     round(float(best_batch["Quality_Score"]), 3),
            "Yield_Score":       round(float(best_batch["Yield_Score"]), 3),
            "Total_Energy_kWh":  round(float(best_batch["Total_Energy_kWh"]), 3),
            "Carbon_kg_CO2":     round(float(best_batch["Carbon_kg_CO2"]), 3),
            "Performance_Score": round(float(best_batch["Performance_Score"]), 3),
            "composite_score":   round(float(composite), 4),
        },

        "source_batch":     best_batch["Batch_ID"],
        "pareto_solutions": len(pareto_idx),
        "pareto_front":     pareto_points,
    }

    return signature

# ─── BUILD ALL 3 ──────────────────────────────────────────
def build_all_signatures():
    print("=" * 55)
    print("   AuraOptima — Golden Signature Engine")
    print("   (Mode-Specific Pareto from Real Batch Data)")
    print("=" * 55)

    signatures = {}
    for mode_key in MODES:
        sig = build_signature(mode_key)
        signatures[mode_key] = sig

        print(f"\n✅ Mode     : {sig['label']}")
        print(f"   Source   : {sig['source_batch']}")
        print(f"   Quality  : {sig['benchmark_scores']['Quality_Score']}")
        print(f"   Yield    : {sig['benchmark_scores']['Yield_Score']}")
        print(f"   Energy   : {sig['benchmark_scores']['Total_Energy_kWh']} kWh")
        print(f"   Carbon   : {sig['benchmark_scores']['Carbon_kg_CO2']} kg CO₂")
        print(f"   Pareto   : {sig['pareto_solutions']} solutions")
        print(f"   Params   :")
        for k, v in sig["optimal_params"].items():
            print(f"      {k:25s}: {v}")

    os.makedirs("data", exist_ok=True)
    with open(SIGNATURES_FILE, "w") as f:
        json.dump(signatures, f, indent=2)

    print(f"\n✅ Saved → {SIGNATURES_FILE}")
    return signatures

# ─── CONTINUOUS LEARNING ──────────────────────────────────
def check_and_propose_update(new_batch_row, signatures):
    proposals = []
    q_max = df["Quality_Score"].max()
    y_max = df["Yield_Score"].max()
    e_max = df["Total_Energy_kWh"].max()

    for mode_key, sig in signatures.items():
        bench   = sig["benchmark_scores"]
        weights = sig["weights"]

        new_q = new_batch_row.get("Quality_Score", 0)
        new_y = new_batch_row.get("Yield_Score", 0)
        new_e = new_batch_row.get("Total_Energy_kWh", 9999)

        new_composite = (
            weights["quality"] * (new_q / q_max) +
            weights["yield"]   * (new_y / y_max) +
            weights["energy"]  * (1 - new_e / e_max)
        )

        if new_composite > bench["composite_score"]:
            proposals.append({
                "mode":        mode_key,
                "batch_id":    new_batch_row.get("Batch_ID"),
                "new_score":   round(new_composite, 4),
                "old_score":   bench["composite_score"],
                "improvement": round((new_composite - bench["composite_score"]) * 100, 2),
                "status":      "PENDING_APPROVAL"
            })

    return proposals

# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    signatures = build_all_signatures()

    print("\n\n📊 Real Data Insights:")
    print("-" * 55)
    print(f"  Best Quality Batch  : {df.loc[df['Quality_Score'].idxmax(),    'Batch_ID']} "
          f"(Score: {df['Quality_Score'].max():.2f})")
    print(f"  Best Yield Batch    : {df.loc[df['Yield_Score'].idxmax(),      'Batch_ID']} "
          f"(Score: {df['Yield_Score'].max():.2f})")
    print(f"  Lowest Energy Batch : {df.loc[df['Total_Energy_kWh'].idxmin(), 'Batch_ID']} "
          f"({df['Total_Energy_kWh'].min():.2f} kWh)")
    print(f"  Best Overall Batch  : {df.loc[df['Performance_Score'].idxmax(),'Batch_ID']} "
          f"(Perf: {df['Performance_Score'].max():.2f})")

    print("\n\n🔍 Continuous Learning Check:")
    best = df.loc[df["Performance_Score"].idxmax()].to_dict()
    proposals = check_and_propose_update(best, signatures)
    if proposals:
        for p in proposals:
            print(f"  ⚠️  Batch {p['batch_id']} beats '{p['mode']}' "
                  f"by +{p['improvement']}% → PENDING APPROVAL")
    else:
        print("  ✅ Current signatures are optimal")

    print("\n🎉 Done!")