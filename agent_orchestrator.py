"""
AuraOptima — Multi-Agent Orchestrator
══════════════════════════════════════════════════════════
Three coordinating agents for Track B:
  1. PredictionAgent  — runs deviation analysis + LLM explanations
  2. GoldenSignatureAgent — manages signature lifecycle + proposals
  3. CarbonAgent — tracks emissions, flags regulatory violations
"""

import json
import os
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────
MASTER_CSV      = "data/master_df.csv"
SIGNATURES_FILE = "data/golden_signatures.json"

# Simulated regulatory targets (India pharma industry)
CARBON_REGULATORY_TARGETS = {
    "max_carbon_per_batch_kg":  80.0,    # kg CO2 per batch
    "max_energy_per_batch_kWh": 95.0,    # kWh per batch
    "annual_reduction_target_pct": 10.0, # % year-over-year
    "emission_factor_kg_per_kWh": 0.82,  # India grid factor
}


# ═══════════════════════════════════════════════════════════
# AGENT 1: PREDICTION AGENT
# ═══════════════════════════════════════════════════════════
class PredictionAgent:
    """
    Runs deviation analysis on a batch and generates
    plain-English SHAP-style feature importance explanations.
    """
    name = "Prediction Agent"
    icon = "🔮"

    def __init__(self):
        from deviation_engine import analyze_deviation, df
        self._analyze = analyze_deviation
        self._df = df

    def run(self, batch_id, mode="balanced"):
        """Analyze a batch and return structured results with feature impacts."""
        report = self._analyze(batch_id, mode=mode)
        if "error" in report:
            return {
                "agent": self.name,
                "icon": self.icon,
                "status": "ERROR",
                "error": report["error"],
            }

        # Calculate SHAP-style feature importance (% contribution to deviation)
        param_analysis = report.get("param_analysis", {})
        total_abs_dev = sum(abs(p["deviation_pct"]) for p in param_analysis.values())

        feature_impacts = []
        for param, info in param_analysis.items():
            abs_dev = abs(info["deviation_pct"])
            contribution = round((abs_dev / total_abs_dev * 100), 1) if total_abs_dev > 0 else 0
            feature_impacts.append({
                "parameter": param,
                "deviation_pct": info["deviation_pct"],
                "severity": info["severity"],
                "contribution_pct": contribution,
                "direction": info["direction"],
                "recommendation": info["recommendation"],
            })

        # Sort by contribution
        feature_impacts.sort(key=lambda x: x["contribution_pct"], reverse=True)

        # Build plain English explanation
        top = feature_impacts[:3]
        explanations = []
        for f in top:
            if f["severity"] != "OK":
                explanations.append(
                    f"{f['parameter'].replace('_', ' ')} contributed {f['contribution_pct']}% "
                    f"to the deviation ({f['direction']}, {f['deviation_pct']:+.1f}%)"
                )

        status = report.get("overall_status", "UNKNOWN")
        plain_english = (
            f"Batch {batch_id} is {status}. "
            + (". ".join(explanations) + "." if explanations else "All parameters are within optimal range.")
        )

        return {
            "agent": self.name,
            "icon": self.icon,
            "status": status,
            "batch_id": batch_id,
            "mode": mode,
            "plain_english": plain_english,
            "feature_impacts": feature_impacts,
            "summary": report.get("summary", {}),
            "savings_potential": report.get("savings_potential", {}),
            "top_actions": report.get("top_actions", []),
            "timestamp": datetime.now().isoformat(),
        }


# ═══════════════════════════════════════════════════════════
# AGENT 2: GOLDEN SIGNATURE AGENT
# ═══════════════════════════════════════════════════════════
class GoldenSignatureAgent:
    """
    Manages golden signature lifecycle.
    Detects if a batch outperforms current signature,
    proposes updates, and calculates projected impact.
    """
    name = "Golden Signature Agent"
    icon = "🧬"

    def __init__(self):
        from golden_signature_engine import check_and_propose_update, df, MODES
        self._check = check_and_propose_update
        self._df = df
        self._modes = MODES

    def run(self, batch_id, mode="balanced"):
        """Check if batch beats current signature and propose update."""
        # Load current signatures
        with open(SIGNATURES_FILE) as f:
            signatures = json.load(f)

        # Get batch data
        batch_row = self._df[self._df["Batch_ID"] == batch_id]
        if batch_row.empty:
            return {
                "agent": self.name,
                "icon": self.icon,
                "status": "ERROR",
                "error": f"Batch {batch_id} not found",
            }

        batch_data = batch_row.iloc[0].to_dict()
        proposals = self._check(batch_data, signatures)

        # Filter to requested mode
        mode_proposals = [p for p in proposals if p["mode"] == mode]
        proposal_created = len(mode_proposals) > 0

        # Calculate projected impact
        sig = signatures.get(mode, {})
        bench = sig.get("benchmark_scores", {})
        projected_impact = {
            "energy_change_kWh": round(batch_data.get("Total_Energy_kWh", 0) - bench.get("Total_Energy_kWh", 0), 2),
            "carbon_change_kg": round(batch_data.get("Carbon_kg_CO2", 0) - bench.get("Carbon_kg_CO2", 0), 2),
            "quality_change": round(batch_data.get("Quality_Score", 0) - bench.get("Quality_Score", 0), 2),
            "yield_change": round(batch_data.get("Yield_Score", 0) - bench.get("Yield_Score", 0), 2),
        }

        # Confidence score based on how much it beats the current signature
        if mode_proposals:
            improvement = mode_proposals[0]["improvement"]
            # Map improvement to confidence: 0-2% = low, 2-5% = medium, 5%+ = high
            if improvement >= 5:
                confidence = {"score": 0.92, "level": "HIGH"}
            elif improvement >= 2:
                confidence = {"score": 0.75, "level": "MEDIUM"}
            else:
                confidence = {"score": 0.58, "level": "LOW"}
        else:
            confidence = {"score": 0.0, "level": "N/A"}

        # Plain English
        if proposal_created:
            imp = mode_proposals[0]["improvement"]
            plain_english = (
                f"Batch {batch_id} outperforms the current '{mode}' golden signature by "
                f"{imp}%. A signature update is proposed. "
                f"Projected energy impact: {projected_impact['energy_change_kWh']:+.1f} kWh, "
                f"carbon: {projected_impact['carbon_change_kg']:+.1f} kg CO₂. "
                f"Confidence: {confidence['level']} ({confidence['score']:.0%})."
            )
            status = "PROPOSAL_CREATED"
        else:
            plain_english = (
                f"Batch {batch_id} does not exceed the current '{mode}' golden signature. "
                f"No update proposed. Current signature remains optimal."
            )
            status = "NO_UPDATE_NEEDED"

        return {
            "agent": self.name,
            "icon": self.icon,
            "status": status,
            "batch_id": batch_id,
            "mode": mode,
            "proposal_created": proposal_created,
            "proposals": mode_proposals,
            "projected_impact": projected_impact,
            "confidence": confidence,
            "plain_english": plain_english,
            "current_signature_version": sig.get("version", 1),
            "current_source_batch": sig.get("source_batch", "N/A"),
            "timestamp": datetime.now().isoformat(),
        }


# ═══════════════════════════════════════════════════════════
# AGENT 3: CARBON AGENT
# ═══════════════════════════════════════════════════════════
class CarbonAgent:
    """
    Tracks batch-level emissions, aligns with regulatory targets,
    and flags deviations.
    """
    name = "Carbon Agent"
    icon = "🌿"

    def __init__(self):
        import pandas as pd
        self._df = pd.read_csv(MASTER_CSV)
        self._targets = CARBON_REGULATORY_TARGETS

    def run(self, batch_id, mode="balanced"):
        """Check carbon compliance for a batch."""
        batch_row = self._df[self._df["Batch_ID"] == batch_id]
        if batch_row.empty:
            return {
                "agent": self.name,
                "icon": self.icon,
                "status": "ERROR",
                "error": f"Batch {batch_id} not found",
            }

        batch = batch_row.iloc[0]
        batch_carbon = float(batch["Carbon_kg_CO2"])
        batch_energy = float(batch["Total_Energy_kWh"])

        max_carbon = self._targets["max_carbon_per_batch_kg"]
        max_energy = self._targets["max_energy_per_batch_kWh"]

        # Check violations
        violations = []
        if batch_carbon > max_carbon:
            violations.append({
                "type": "CARBON_EXCEEDS_LIMIT",
                "severity": "CRITICAL" if batch_carbon > max_carbon * 1.2 else "WARNING",
                "message": f"Carbon {batch_carbon:.1f} kg exceeds limit of {max_carbon:.0f} kg ({((batch_carbon/max_carbon - 1)*100):.1f}% over)",
                "actual": batch_carbon,
                "target": max_carbon,
                "overshoot_pct": round((batch_carbon / max_carbon - 1) * 100, 1),
            })
        if batch_energy > max_energy:
            violations.append({
                "type": "ENERGY_EXCEEDS_LIMIT",
                "severity": "CRITICAL" if batch_energy > max_energy * 1.2 else "WARNING",
                "message": f"Energy {batch_energy:.1f} kWh exceeds limit of {max_energy:.0f} kWh ({((batch_energy/max_energy - 1)*100):.1f}% over)",
                "actual": batch_energy,
                "target": max_energy,
                "overshoot_pct": round((batch_energy / max_energy - 1) * 100, 1),
            })

        compliant = len(violations) == 0

        # Fleet context
        fleet_avg_carbon = float(self._df["Carbon_kg_CO2"].mean())
        fleet_avg_energy = float(self._df["Total_Energy_kWh"].mean())
        fleet_non_compliant = len(self._df[
            (self._df["Carbon_kg_CO2"] > max_carbon) |
            (self._df["Total_Energy_kWh"] > max_energy)
        ])

        # SDG alignment
        sdg_impact = {
            "SDG_7": "Affordable & Clean Energy — " + ("On track ✅" if batch_energy <= max_energy else "At risk ⚠️"),
            "SDG_13": "Climate Action — " + ("On track ✅" if batch_carbon <= max_carbon else "At risk ⚠️"),
        }

        # Plain English
        if compliant:
            plain_english = (
                f"Batch {batch_id} is COMPLIANT. Carbon: {batch_carbon:.1f} kg (limit: {max_carbon:.0f} kg), "
                f"Energy: {batch_energy:.1f} kWh (limit: {max_energy:.0f} kWh). "
                f"No regulatory violations detected."
            )
            status = "COMPLIANT"
        else:
            violation_msgs = [v["message"] for v in violations]
            plain_english = (
                f"Batch {batch_id} has {len(violations)} regulatory violation(s): "
                + "; ".join(violation_msgs) + ". "
                f"Immediate action recommended."
            )
            status = "NON_COMPLIANT"

        return {
            "agent": self.name,
            "icon": self.icon,
            "status": status,
            "batch_id": batch_id,
            "compliant": compliant,
            "batch_carbon_kg": round(batch_carbon, 2),
            "batch_energy_kWh": round(batch_energy, 2),
            "violations": violations,
            "regulatory_targets": self._targets,
            "fleet_context": {
                "avg_carbon_kg": round(fleet_avg_carbon, 2),
                "avg_energy_kWh": round(fleet_avg_energy, 2),
                "non_compliant_batches": fleet_non_compliant,
                "total_batches": len(self._df),
                "compliance_rate_pct": round((1 - fleet_non_compliant / len(self._df)) * 100, 1),
            },
            "sdg_alignment": sdg_impact,
            "plain_english": plain_english,
            "timestamp": datetime.now().isoformat(),
        }


# ═══════════════════════════════════════════════════════════
# ORCHESTRATOR — Runs all agents and collects results
# ═══════════════════════════════════════════════════════════
def run_all_agents(batch_id, mode="balanced"):
    """
    Run all 3 agents for a batch and return combined results.
    This is the main entry point for the multi-agent system.
    """
    results = {
        "batch_id": batch_id,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "agents": [],
        "notifications": [],
    }

    # Agent 1: Prediction
    try:
        pred = PredictionAgent()
        pred_result = pred.run(batch_id, mode)
        results["agents"].append(pred_result)
        if pred_result.get("status") in ("WARNING", "CRITICAL"):
            results["notifications"].append({
                "agent": pred.name,
                "icon": pred.icon,
                "type": pred_result["status"],
                "message": pred_result["plain_english"],
            })
    except Exception as e:
        results["agents"].append({"agent": "Prediction Agent", "icon": "🔮", "status": "ERROR", "error": str(e)})

    # Agent 2: Golden Signature
    try:
        gs = GoldenSignatureAgent()
        gs_result = gs.run(batch_id, mode)
        results["agents"].append(gs_result)
        if gs_result.get("proposal_created"):
            results["notifications"].append({
                "agent": gs.name,
                "icon": gs.icon,
                "type": "PROPOSAL",
                "message": gs_result["plain_english"],
            })
    except Exception as e:
        results["agents"].append({"agent": "Golden Signature Agent", "icon": "🧬", "status": "ERROR", "error": str(e)})

    # Agent 3: Carbon
    try:
        carbon = CarbonAgent()
        carbon_result = carbon.run(batch_id, mode)
        results["agents"].append(carbon_result)
        if not carbon_result.get("compliant"):
            results["notifications"].append({
                "agent": carbon.name,
                "icon": carbon.icon,
                "type": "VIOLATION",
                "message": carbon_result["plain_english"],
            })
    except Exception as e:
        results["agents"].append({"agent": "Carbon Agent", "icon": "🌿", "status": "ERROR", "error": str(e)})

    # Overall summary
    statuses = [a.get("status", "UNKNOWN") for a in results["agents"]]
    if "CRITICAL" in statuses or "NON_COMPLIANT" in statuses:
        results["overall_status"] = "ACTION_REQUIRED"
    elif "WARNING" in statuses or "PROPOSAL_CREATED" in statuses:
        results["overall_status"] = "REVIEW_NEEDED"
    else:
        results["overall_status"] = "ALL_CLEAR"

    results["notification_count"] = len(results["notifications"])

    return results


def get_agent_notifications(top_n=5):
    """
    Scan fleet for agent notifications — checks the worst-performing batches
    and returns all alerts from all agents.
    """
    import pandas as pd
    df = pd.read_csv(MASTER_CSV)

    # Pick the top_n worst batches by performance score
    worst = df.nsmallest(top_n, "Performance_Score")
    all_notifications = []

    for _, row in worst.iterrows():
        bid = row["Batch_ID"]
        try:
            carbon = CarbonAgent()
            result = carbon.run(bid)
            if not result.get("compliant"):
                all_notifications.append({
                    "agent": carbon.name,
                    "icon": carbon.icon,
                    "batch_id": bid,
                    "type": "VIOLATION",
                    "severity": "CRITICAL" if any(v["severity"] == "CRITICAL" for v in result.get("violations", [])) else "WARNING",
                    "message": result.get("plain_english", ""),
                })
        except Exception:
            pass

        # Check prediction status
        try:
            pred = PredictionAgent()
            result = pred.run(bid)
            if result.get("status") in ("WARNING", "CRITICAL"):
                all_notifications.append({
                    "agent": pred.name,
                    "icon": pred.icon,
                    "batch_id": bid,
                    "type": result["status"],
                    "severity": result["status"],
                    "message": result.get("plain_english", ""),
                })
        except Exception:
            pass

    return {
        "notifications": all_notifications,
        "count": len(all_notifications),
        "batches_scanned": top_n,
    }


# ─── MAIN DEMO ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   AuraOptima — Multi-Agent Orchestrator")
    print("=" * 55)

    for bid in ["T001", "T038", "T046"]:
        print(f"\n{'─' * 55}")
        print(f"Running all agents for Batch {bid}...")
        result = run_all_agents(bid, "balanced")

        print(f"\n  Overall: {result['overall_status']}")
        print(f"  Notifications: {result['notification_count']}")

        for agent in result["agents"]:
            print(f"\n  {agent.get('icon', '?')} {agent.get('agent', '?')} → {agent.get('status', '?')}")
            if "plain_english" in agent:
                print(f"     {agent['plain_english'][:120]}...")

    print("\n🎉 Orchestrator complete!")
