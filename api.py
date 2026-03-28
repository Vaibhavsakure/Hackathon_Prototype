from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import json, os, sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # type: ignore

sys.path.insert(0, os.path.dirname(__file__))
from deviation_engine import analyze_deviation, rank_all_batches
from hitl_manager import (
    propose_update, accept_proposal, reject_proposal,
    reprioritize_mode, get_pending_proposals,
    get_decisions_summary, load_json, save_json
)
from llm_assistant import chat as llm_chat, generate_batch_insight, generate_why_why_analysis
from agent_orchestrator import run_all_agents, CarbonAgent, get_agent_notifications
from knowledge_graph import build_graph, get_graph_summary, get_visualization_graph, get_node_relationships, query_path, get_natural_language_summary
from report_generator import generate_batch_report
from database import (
    init_db, get_batches_dataframe, get_all_batches,
    get_pending_proposals_db, get_decisions_summary_db,
    get_db_stats
)
import pandas as pd

app = FastAPI(title="AuraOptima API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MASTER_CSV      = "data/master_df.csv"
SIGNATURES_FILE = "data/golden_signatures.json"

# ── Initialize SQLite database on startup ─────────────────
init_db()

def load_df():
    """Load batch data from SQLite database."""
    return get_batches_dataframe()

# ── GET /batches ──────────────────────────────────────────
@app.get("/batches")
def get_batches():
    return get_all_batches()

# ── GET /golden-signatures ────────────────────────────────
@app.get("/golden-signatures")
def get_signatures():
    sigs = load_json(SIGNATURES_FILE, {})
    return sigs

# ── GET /deviation/{batch_id} ─────────────────────────────
@app.get("/deviation/{batch_id}")
def get_deviation(batch_id: str, mode: str = "balanced"):
    report = analyze_deviation(batch_id, mode=mode)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])
    return report

# ── GET /rankings ─────────────────────────────────────────
@app.get("/rankings")
def get_rankings(mode: str = "balanced"):
    return rank_all_batches(mode)

# ── POST /optimize ────────────────────────────────────────
class OptimizeRequest(BaseModel):
    mode: str = "balanced"

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    sigs = load_json(SIGNATURES_FILE, {})
    if req.mode not in sigs:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")
    sig = sigs[req.mode]
    return {
        "mode":           req.mode,
        "label":          sig["label"],
        "optimal_params": sig["optimal_params"],
        "benchmark":      sig["benchmark_scores"],
        "source_batch":   sig["source_batch"],
        "pareto_front":   sig.get("pareto_front", []),
        "pareto_count":   sig.get("pareto_solutions", 0),
    }

# ── GET /proposals ────────────────────────────────────────
@app.get("/proposals")
def get_proposals():
    return get_pending_proposals_db()

# ── POST /approve-update ──────────────────────────────────
class ApproveRequest(BaseModel):
    proposal_id: str
    action: str          # "accept" or "reject"
    reviewer_note: Optional[str] = ""

@app.post("/approve-update")
def approve_update(req: ApproveRequest):
    if req.action == "accept":
        ok = accept_proposal(req.proposal_id, req.reviewer_note or "")
    elif req.action == "reject":
        ok = reject_proposal(req.proposal_id, req.reviewer_note or "")
    else:
        raise HTTPException(status_code=400, detail="action must be 'accept' or 'reject'")
    if not ok:
        raise HTTPException(status_code=404, detail="Proposal not found or already reviewed")
    return {"status": "ok", "action": req.action, "proposal_id": req.proposal_id}

# ── GET /decisions ────────────────────────────────────────
@app.get("/decisions")
def get_decisions():
    return get_decisions_summary_db()

# ── GET /sustainability ───────────────────────────────────
@app.get("/sustainability")
def get_sustainability():
    df = load_df()
    sigs = load_json(SIGNATURES_FILE, {})
    golden_energy = sigs["balanced"]["benchmark_scores"]["Total_Energy_kWh"]
    golden_carbon = sigs["balanced"]["benchmark_scores"]["Carbon_kg_CO2"]

    total_energy = float(df["Total_Energy_kWh"].sum())
    total_carbon = float(df["Carbon_kg_CO2"].sum())
    avg_energy   = float(df["Total_Energy_kWh"].mean())
    avg_carbon   = float(df["Carbon_kg_CO2"].mean())

    potential_energy_savings = max(0, (avg_energy - golden_energy) * len(df))
    potential_carbon_savings = max(0, (avg_carbon - golden_carbon) * len(df))

    trend = df[["Batch_ID","Total_Energy_kWh","Carbon_kg_CO2","Quality_Score","Performance_Score"]].to_dict("records")

    return {
        "total_energy_kWh":        round(total_energy, 2),
        "total_carbon_kg":         round(total_carbon, 2),
        "avg_energy_kWh":          round(avg_energy, 2),
        "avg_carbon_kg":           round(avg_carbon, 2),
        "golden_energy_kWh":       round(golden_energy, 2),
        "golden_carbon_kg":        round(golden_carbon, 2),
        "potential_energy_savings": round(potential_energy_savings, 2),
        "potential_carbon_savings": round(potential_carbon_savings, 2),
        "batch_trend":             trend,
    }

# ── POST /chat ────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        history = [m.model_dump() for m in req.history] if req.history else []
        response = llm_chat(req.message, conversation_history=history)
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

# ── GET /insights/{batch_id} ──────────────────────────────
@app.get("/insights/{batch_id}")
def get_insights(batch_id: str, mode: str = "balanced"):
    try:
        result = generate_batch_insight(batch_id, mode=mode)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

# ── GET /report/{batch_id} (PDF Download) ─────────────────
@app.get("/report/{batch_id}")
def download_report(batch_id: str, mode: str = "balanced"):
    try:
        report = analyze_deviation(batch_id, mode=mode)
        if "error" in report:
            raise HTTPException(status_code=404, detail=report["error"])

        # Get full batch data for the report
        df = load_df()
        batch_row = df[df["Batch_ID"] == batch_id]
        batch_data = batch_row.iloc[0].to_dict() if not batch_row.empty else None

        pdf_bytes = generate_batch_report(report, batch_data=batch_data, mode=mode)

        filename = f"AuraOptima_Report_{batch_id}_{mode}.pdf"
        return Response(
            content=bytes(pdf_bytes) if pdf_bytes is not None else b"",
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition",
                "Content-Type": "application/pdf"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

# ── GET /fleet-health ─────────────────────────────────────
@app.get("/fleet-health")
def get_fleet_health():
    df = load_df()
    sigs = load_json(SIGNATURES_FILE, {})
    golden = sigs["balanced"]["optimal_params"]
    bench = sigs["balanced"]["benchmark_scores"]

    decision_vars = [
        "Granulation_Time", "Binder_Amount", "Drying_Temp",
        "Drying_Time", "Compression_Force", "Machine_Speed", "Lubricant_Conc"
    ]

    healthy = 0
    warning = 0
    critical = 0
    attention_batches = []

    for _, row in df.iterrows():
        crit_count = 0
        warn_count = 0
        for var in decision_vars:
            gval = float(golden[var])
            bval = float(row[var])
            if gval != 0:
                dev = abs((bval - gval) / gval * 100)
            else:
                dev = 0
            if dev >= 20:
                crit_count += 1
            elif dev >= 10:
                warn_count += 1

        if crit_count >= 3:
            status = "CRITICAL"
            critical += 1
        elif crit_count >= 1 or warn_count >= 3:
            status = "WARNING"
            warning += 1
        else:
            status = "HEALTHY"
            healthy += 1

        if status != "HEALTHY":
            energy_gap = float(row["Total_Energy_kWh"]) - bench["Total_Energy_kWh"]
            attention_batches.append({
                "batch_id": row["Batch_ID"],
                "status": status,
                "quality": round(float(row["Quality_Score"]), 1),
                "energy": round(float(row["Total_Energy_kWh"]), 1),
                "perf": round(float(row["Performance_Score"]), 1),
                "energy_gap": round(max(energy_gap, 0), 1),
            })

    total = len(df)
    health_pct = round((healthy / total) * 100, 1)
    attention_batches.sort(key=lambda x: x["energy_gap"], reverse=True)

    return {
        "total": total,
        "healthy": healthy,
        "warning": warning,
        "critical": critical,
        "health_pct": health_pct,
        "attention_batches": attention_batches[:5],
    }


# ── POST /simulate ────────────────────────────────────────
class SimulateRequest(BaseModel):
    params: dict
    mode: str = "balanced"

@app.post("/simulate")
def simulate(req: SimulateRequest):
    """
    Predict outcomes for hypothetical parameter values using
    linear interpolation from the dataset.
    """

    df = load_df()
    sigs = load_json(SIGNATURES_FILE, {})
    sig = sigs.get(req.mode, sigs["balanced"])
    golden = sig["optimal_params"]
    bench = sig["benchmark_scores"]

    decision_vars = [
        "Granulation_Time", "Binder_Amount", "Drying_Temp",
        "Drying_Time", "Compression_Force", "Machine_Speed", "Lubricant_Conc"
    ]
    outcomes = ["Quality_Score", "Yield_Score", "Total_Energy_kWh", "Carbon_kg_CO2", "Performance_Score"]

    X = df[decision_vars].values
    predictions = {}

    # Clamp input parameters to dataset min/max to prevent wild extrapolation
    clamped_inputs = []
    for v in decision_vars:
        val = float(req.params.get(v, golden[v]))
        val = max(float(df[v].min()), min(float(df[v].max()), val))
        clamped_inputs.append(val)
    input_vals = np.array([clamped_inputs])

    # Realistic bounds for outcome clamping
    outcome_bounds = {
        "Quality_Score":     (0, 100),
        "Yield_Score":       (0, 100),
        "Total_Energy_kWh":  (0, float(df["Total_Energy_kWh"].max()) * 1.5),
        "Carbon_kg_CO2":     (0, float(df["Carbon_kg_CO2"].max()) * 1.5),
        "Performance_Score": (0, 100),
    }

    for outcome in outcomes:
        y = df[outcome].values
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=6).fit(X, y)
        pred = float(model.predict(input_vals)[0])
        lo, hi = outcome_bounds.get(outcome, (0, 9999))
        pred = max(lo, min(hi, pred))  # clamp to realistic range
        predictions[outcome] = round(pred, 2)

    # Param ranges for slider bounds
    param_ranges = {}
    for var in decision_vars:
        param_ranges[var] = {
            "min": round(float(df[var].min()), 2),
            "max": round(float(df[var].max()), 2),
            "golden": float(golden[var]),
            "step": round((float(df[var].max()) - float(df[var].min())) / 50, 2),
        }

    # Compare vs golden
    comparison = {}
    for k in outcomes:
        golden_val = bench.get(k, predictions[k])
        diff = predictions[k] - golden_val
        higher_better = k not in ["Total_Energy_kWh", "Carbon_kg_CO2"]
        good = diff >= 0 if higher_better else diff <= 0
        comparison[k] = {
            "predicted": predictions[k],
            "golden": round(golden_val, 2),
            "diff": round(diff, 2),
            "good": good,
        }

    return {
        "predictions": predictions,
        "comparison": comparison,
        "param_ranges": param_ranges,
        "mode": req.mode,
    }


# ── GET /roi ──────────────────────────────────────────────
@app.get("/roi")
def get_roi(cost_per_kwh: float = 8.0):
    """
    Calculate ROI / cost savings if all underperforming batches
    were optimized to match the golden signature.
    """
    df = load_df()
    sigs = load_json(SIGNATURES_FILE, {})
    bench = sigs["balanced"]["benchmark_scores"]
    golden_energy = bench["Total_Energy_kWh"]
    golden_carbon = bench["Carbon_kg_CO2"]

    batch_savings = []
    total_energy_saved = 0
    total_carbon_saved = 0
    improvable_count = 0

    for _, row in df.iterrows():
        e_gap = float(row["Total_Energy_kWh"]) - golden_energy
        c_gap = float(row["Carbon_kg_CO2"]) - golden_carbon
        if e_gap > 0:
            improvable_count += 1
            total_energy_saved += e_gap
            total_carbon_saved += c_gap
            batch_savings.append({
                "batch_id": row["Batch_ID"],
                "energy_saved_kwh": round(e_gap, 2),
                "carbon_saved_kg": round(c_gap, 2),
                "cost_saved": round(e_gap * cost_per_kwh, 2),
                "current_energy": round(float(row["Total_Energy_kWh"]), 2),
                "quality": round(float(row["Quality_Score"]), 1),
            })

    batch_savings.sort(key=lambda x: x["energy_saved_kwh"], reverse=True)

    # Annual projections (assume 250 production days, 1 batch/day per line)
    batches_per_year = 250
    annual_energy_saved = total_energy_saved * (batches_per_year / len(df))
    annual_carbon_saved = total_carbon_saved * (batches_per_year / len(df))
    annual_cost_saved = annual_energy_saved * cost_per_kwh

    return {
        "cost_per_kwh": cost_per_kwh,
        "currency": "INR",
        "improvable_batches": improvable_count,
        "total_batches": len(df),
        "total_energy_saved_kwh": round(total_energy_saved, 2),
        "total_carbon_saved_kg": round(total_carbon_saved, 2),
        "total_cost_saved": round(total_energy_saved * cost_per_kwh, 2),
        "annual_energy_saved_kwh": round(annual_energy_saved, 2),
        "annual_carbon_saved_kg": round(annual_carbon_saved, 2),
        "annual_cost_saved": round(annual_cost_saved, 2),
        "annual_carbon_saved_tonnes": round(annual_carbon_saved / 1000, 3),
        "batch_savings": batch_savings[:15],
    }


# ── GET /knowledge-graph ──────────────────────────────────
@app.get("/knowledge-graph")
def get_knowledge_graph(full: bool = False):
    """Get knowledge graph nodes and edges."""
    if full:
        return build_graph()
    return get_visualization_graph()

@app.get("/knowledge-graph/summary")
def knowledge_graph_summary():
    """Get knowledge graph statistics."""
    return get_graph_summary()

@app.get("/knowledge-graph/node/{node_id}")
def knowledge_graph_node(node_id: str):
    """Get relationships for a specific node."""
    graph = build_graph()
    rels = get_node_relationships(graph, node_id)
    node = next((n for n in graph["nodes"] if n["id"] == node_id), None)
    return {"node": node, "relationships": rels[:30]}

@app.get("/knowledge-graph/path")
def knowledge_graph_path(from_node: str, to_node: str):
    """Multi-hop path query between two nodes."""
    return query_path(from_node, to_node)

@app.get("/knowledge-graph/nl-summary/{node_id}")
def knowledge_graph_nl_summary(node_id: str):
    """Get natural language summary for a node."""
    return get_natural_language_summary(node_id)

@app.get("/agents/notifications")
def agent_notifications():
    """Get all agent notifications from fleet scan."""
    return get_agent_notifications()

# ── GET /agents/run/{batch_id} ────────────────────────────
@app.get("/agents/run/{batch_id}")
def run_agents(batch_id: str, mode: str = "balanced"):
    """Run all 3 agents for a batch and return combined results."""
    try:
        return run_all_agents(batch_id, mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

# ── GET /carbon-compliance/{batch_id} ─────────────────────
@app.get("/carbon-compliance/{batch_id}")
def carbon_compliance(batch_id: str):
    """Check carbon regulatory compliance for a batch."""
    try:
        agent = CarbonAgent()
        return agent.run(batch_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Carbon agent error: {str(e)}")

# ── GET /why-why/{batch_id} ───────────────────────────────
@app.get("/why-why/{batch_id}")
def why_why_analysis(batch_id: str, mode: str = "balanced"):
    """Generate why-why root cause analysis for a batch."""
    try:
        result = generate_why_why_analysis(batch_id, mode)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Why-why analysis error: {str(e)}")

# ── GET /db-stats ─────────────────────────────────────────
@app.get("/db-stats")
def db_stats():
    """Database statistics for demo/judges."""
    return get_db_stats()

# ── GET / (serve frontend) ────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

# Entry point moved to app.py
# Run: python app.py