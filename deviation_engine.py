import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────
MASTER_CSV      = "data/master_df.csv"
SIGNATURES_FILE = "data/golden_signatures.json"
RECOMMENDATIONS_FILE = "data/recommendations.json"

DECISION_VARS = [
    "Granulation_Time", "Binder_Amount", "Drying_Temp",
    "Drying_Time", "Compression_Force", "Machine_Speed", "Lubricant_Conc"
]

# Thresholds for deviation alerts
THRESHOLD_WARNING  = 10.0   # % deviation → yellow
THRESHOLD_CRITICAL = 20.0   # % deviation → red

# ─── LOAD DATA ────────────────────────────────────────────
df         = pd.read_csv(MASTER_CSV)
with open(SIGNATURES_FILE) as f:
    signatures = json.load(f)

# ─── CORE: DEVIATION ANALYSIS ─────────────────────────────
def analyze_deviation(batch_id, mode="balanced"):
    """
    Compare a batch's parameters against the golden signature for given mode.
    Returns full deviation report with recommendations.
    """
    # Get batch row
    batch = df[df["Batch_ID"] == batch_id]
    if batch.empty:
        return {"error": f"Batch {batch_id} not found"}
    batch = batch.iloc[0]

    sig         = signatures[mode]
    golden      = sig["optimal_params"]
    bench       = sig["benchmark_scores"]

    # ── Parameter Deviation Analysis ──
    param_analysis = {}
    for var in DECISION_VARS:
        batch_val  = float(batch[var])
        golden_val = float(golden[var])

        if golden_val != 0:
            deviation_pct = ((batch_val - golden_val) / golden_val) * 100
        else:
            deviation_pct = 0.0

        # Severity
        abs_dev = abs(deviation_pct)
        if abs_dev >= THRESHOLD_CRITICAL:
            severity = "CRITICAL"
            color    = "red"
        elif abs_dev >= THRESHOLD_WARNING:
            severity = "WARNING"
            color    = "orange"
        else:
            severity = "OK"
            color    = "green"

        # Direction
        direction = "HIGH" if deviation_pct > 0 else "LOW" if deviation_pct < 0 else "ON TARGET"

        # Recommendation
        if severity != "OK":
            diff = golden_val - batch_val
            action = f"{'Increase' if diff > 0 else 'Decrease'} by {abs(diff):.1f}"
            rec    = f"{action} (current: {batch_val:.1f}, target: {golden_val:.1f})"
        else:
            rec = "Within optimal range ✓"

        param_analysis[var] = {
            "batch_value":     round(batch_val, 3),
            "golden_value":    round(golden_val, 3),
            "deviation_pct":   round(deviation_pct, 2),
            "severity":        severity,
            "color":           color,
            "direction":       direction,
            "recommendation":  rec,
        }

    # ── Outcome Comparison ──
    outcome_comparison = {
        "Quality_Score": {
            "batch":  round(float(batch["Quality_Score"]), 3),
            "golden": bench["Quality_Score"],
            "gap":    round(float(batch["Quality_Score"]) - bench["Quality_Score"], 3),
            "status": "✅ Above" if batch["Quality_Score"] >= bench["Quality_Score"] else "⚠️ Below"
        },
        "Yield_Score": {
            "batch":  round(float(batch["Yield_Score"]), 3),
            "golden": bench["Yield_Score"],
            "gap":    round(float(batch["Yield_Score"]) - bench["Yield_Score"], 3),
            "status": "✅ Above" if batch["Yield_Score"] >= bench["Yield_Score"] else "⚠️ Below"
        },
        "Total_Energy_kWh": {
            "batch":  round(float(batch["Total_Energy_kWh"]), 3),
            "golden": bench["Total_Energy_kWh"],
            "gap":    round(float(batch["Total_Energy_kWh"]) - bench["Total_Energy_kWh"], 3),
            "status": "✅ Better" if batch["Total_Energy_kWh"] <= bench["Total_Energy_kWh"] else "⚠️ Higher"
        },
        "Carbon_kg_CO2": {
            "batch":  round(float(batch["Carbon_kg_CO2"]), 3),
            "golden": bench["Carbon_kg_CO2"],
            "gap":    round(float(batch["Carbon_kg_CO2"]) - bench["Carbon_kg_CO2"], 3),
            "status": "✅ Better" if batch["Carbon_kg_CO2"] <= bench["Carbon_kg_CO2"] else "⚠️ Higher"
        },
    }

    # ── Overall Health Score ──
    critical_count = sum(1 for v in param_analysis.values() if v["severity"] == "CRITICAL")
    warning_count  = sum(1 for v in param_analysis.values() if v["severity"] == "WARNING")
    ok_count       = sum(1 for v in param_analysis.values() if v["severity"] == "OK")

    if critical_count >= 3:
        overall_status = "CRITICAL"
    elif critical_count >= 1 or warning_count >= 3:
        overall_status = "WARNING"
    else:
        overall_status = "HEALTHY"

    # ── Top 3 Action Items ──
    sorted_params = sorted(
        param_analysis.items(),
        key=lambda x: abs(x[1]["deviation_pct"]),
        reverse=True
    )
    top_actions = []
    for param, info in sorted_params[:3]:
        if info["severity"] != "OK":
            top_actions.append({
                "parameter":      param,
                "action":         info["recommendation"],
                "deviation":      info["deviation_pct"],
                "severity":       info["severity"]
            })

    # ── Energy Savings Potential ──
    energy_gap     = float(batch["Total_Energy_kWh"]) - bench["Total_Energy_kWh"]
    carbon_gap     = float(batch["Carbon_kg_CO2"])    - bench["Carbon_kg_CO2"]
    savings_potential = {
        "energy_savings_kWh": round(max(energy_gap, 0), 3),
        "carbon_savings_kg":  round(max(carbon_gap, 0), 3),
        "message": (
            f"Aligning to golden signature saves {max(energy_gap,0):.2f} kWh "
            f"and {max(carbon_gap,0):.2f} kg CO₂ per batch"
            if energy_gap > 0
            else "This batch is more energy-efficient than the golden signature!"
        )
    }

    report = {
        "batch_id":           batch_id,
        "mode":               mode,
        "mode_label":         sig["label"],
        "golden_source":      sig["source_batch"],
        "timestamp":          datetime.now().isoformat(),
        "overall_status":     overall_status,
        "summary": {
            "critical_params": critical_count,
            "warning_params":  warning_count,
            "ok_params":       ok_count,
        },
        "param_analysis":     param_analysis,
        "outcome_comparison": outcome_comparison,
        "top_actions":        top_actions,
        "savings_potential":  savings_potential,
    }

    return report

# ─── BATCH RANKING ────────────────────────────────────────
def rank_all_batches(mode="balanced"):
    """Rank all 60 batches by their composite deviation from golden signature."""
    sig    = signatures[mode]
    golden = sig["optimal_params"]
    ranks  = []

    for _, row in df.iterrows():
        total_dev = 0
        for var in DECISION_VARS:
            gval = float(golden[var])
            bval = float(row[var])
            if gval != 0:
                total_dev += abs((bval - gval) / gval * 100)

        avg_dev = total_dev / len(DECISION_VARS)
        ranks.append({
            "Batch_ID":       row["Batch_ID"],
            "avg_deviation":  round(avg_dev, 2),
            "Quality_Score":  round(row["Quality_Score"], 2),
            "Yield_Score":    round(row["Yield_Score"], 2),
            "Energy_kWh":     round(row["Total_Energy_kWh"], 2),
            "Carbon_kg":      round(row["Carbon_kg_CO2"], 2),
            "Perf_Score":     round(row["Performance_Score"], 2),
            "similarity_pct": round(max(0, 100 - avg_dev), 2)
        })

    ranked = sorted(ranks, key=lambda x: x["avg_deviation"])
    return ranked

# ─── SAVE ALL RECOMMENDATIONS ─────────────────────────────
def generate_all_recommendations():
    """Generate and save deviation reports for all batches in balanced mode."""
    all_reports = {}
    for batch_id in df["Batch_ID"].tolist():
        all_reports[batch_id] = analyze_deviation(batch_id, mode="balanced")

    os.makedirs("data", exist_ok=True)
    with open(RECOMMENDATIONS_FILE, "w") as f:
        json.dump(all_reports, f, indent=2)

    print(f"✅ Saved recommendations for {len(all_reports)} batches → {RECOMMENDATIONS_FILE}")
    return all_reports

# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   AuraOptima — Deviation Engine")
    print("=" * 55)

    # Demo: analyze 3 sample batches
    sample_batches = ["T001", "T038", "T046"]
    for bid in sample_batches:
        report = analyze_deviation(bid, mode="balanced")
        print(f"\n📋 Batch {bid} vs Golden Signature ({report['mode_label']})")
        print(f"   Overall Status  : {report['overall_status']}")
        print(f"   Critical Params : {report['summary']['critical_params']}")
        print(f"   Warning Params  : {report['summary']['warning_params']}")
        print(f"   OK Params       : {report['summary']['ok_params']}")
        print(f"   Energy Savings  : {report['savings_potential']['energy_savings_kWh']} kWh")
        print(f"   Top Actions:")
        for action in report["top_actions"]:
            print(f"      [{action['severity']}] {action['parameter']}: {action['action']}")

    # Rank all batches
    print("\n\n🏆 Top 5 Batches (closest to balanced golden signature):")
    print("-" * 55)
    ranked = rank_all_batches("balanced")
    for r in ranked[:5]:
        print(f"   {r['Batch_ID']} | Similarity: {r['similarity_pct']}% | "
              f"Quality: {r['Quality_Score']} | Energy: {r['Energy_kWh']} kWh")

    print("\n\n⚠️  Bottom 5 Batches (furthest from golden signature):")
    print("-" * 55)
    for r in ranked[-5:]:
        print(f"   {r['Batch_ID']} | Similarity: {r['similarity_pct']}% | "
              f"Quality: {r['Quality_Score']} | Energy: {r['Energy_kWh']} kWh")

    # Generate all recommendations
    print("\n")
    generate_all_recommendations()
    print("\n🎉 Deviation Engine complete!")