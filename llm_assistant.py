"""
AuraOptima — LLM Assistant Module
Uses local Ollama (Llama 3.2) to provide natural language insights
and conversational Q&A about batch manufacturing data.
"""

import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads .env file automatically

# ─── CONFIG ───────────────────────────────────────────────
MASTER_CSV      = "data/master_df.csv"
SIGNATURES_FILE = "data/golden_signatures.json"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")

# ─── LOAD HELPERS ─────────────────────────────────────────
def _load_signatures():
    with open(SIGNATURES_FILE) as f:
        return json.load(f)

def _load_df():
    return pd.read_csv(MASTER_CSV)

# ─── OLLAMA API HELPER ────────────────────────────────────
def _ollama_generate(prompt, system=None, max_tokens=1024):
    """Call Ollama's local chat API."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if isinstance(prompt, list):
        messages.extend(prompt)
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    except requests.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. Make sure Ollama is running "
            "(run 'ollama serve' in a terminal)."
        )
    except Exception as e:
        raise RuntimeError(f"Ollama error: {str(e)}")


# ─── SYSTEM PROMPT ────────────────────────────────────────
def build_system_prompt():
    """Build a rich system prompt with domain context."""
    sigs = _load_signatures()
    df   = _load_df()

    # Fleet summary stats
    fleet_stats = {
        "total_batches":     len(df),
        "avg_quality":       round(df["Quality_Score"].mean(), 2),
        "avg_yield":         round(df["Yield_Score"].mean(), 2),
        "avg_energy_kWh":    round(df["Total_Energy_kWh"].mean(), 2),
        "avg_carbon_kg":     round(df["Carbon_kg_CO2"].mean(), 2),
        "avg_performance":   round(df["Performance_Score"].mean(), 2),
        "best_quality_batch":  df.loc[df["Quality_Score"].idxmax(), "Batch_ID"],
        "best_quality_score":  round(df["Quality_Score"].max(), 2),
        "best_energy_batch":   df.loc[df["Total_Energy_kWh"].idxmin(), "Batch_ID"],
        "best_energy_value":   round(df["Total_Energy_kWh"].min(), 2),
        "best_perf_batch":     df.loc[df["Performance_Score"].idxmax(), "Batch_ID"],
        "best_perf_score":     round(df["Performance_Score"].max(), 2),
        "total_energy_kWh":    round(df["Total_Energy_kWh"].sum(), 2),
        "total_carbon_kg":     round(df["Carbon_kg_CO2"].sum(), 2),
    }

    # Golden signature summaries
    sig_summaries = {}
    for mode, sig in sigs.items():
        sig_summaries[mode] = {
            "label":          sig["label"],
            "source_batch":   sig["source_batch"],
            "optimal_params": sig["optimal_params"],
            "benchmark":      sig["benchmark_scores"],
            "weights":        sig["weights"],
        }

    system_prompt = f"""You are AuraOptima AI, an expert pharmaceutical batch manufacturing optimization assistant.

You help engineers and operators understand their batch production data, identify quality issues, optimize energy consumption, and reduce carbon emissions.

## Your Knowledge Base

### System Overview
AuraOptima uses a "Golden Signature" approach — the best-performing batches are identified through multi-objective Pareto analysis and used as benchmarks. Every batch is compared against these golden signatures to find deviations and generate improvement recommendations.

### 7 Decision Variables (controllable parameters):
1. Granulation_Time (minutes)
2. Binder_Amount (grams)
3. Drying_Temp (°C)
4. Drying_Time (minutes)
5. Compression_Force (kN)
6. Machine_Speed (RPM)
7. Lubricant_Conc (%)

### 4 Outcome Metrics:
- Quality_Score (0–100): Composite of Hardness, Dissolution Rate, Friability, Content Uniformity
- Yield_Score (0–100): Based on Moisture Content and Friability (lower = better)
- Total_Energy_kWh: Total energy consumed per batch
- Carbon_kg_CO2: Carbon footprint (Energy × 0.82 India emission factor)
- Performance_Score: 0.4×Quality + 0.4×Yield + 20×EnergyEfficiency

### 3 Optimization Modes:
{json.dumps(sig_summaries, indent=2)}

### Fleet Statistics (60 batches):
{json.dumps(fleet_stats, indent=2)}

## Response Guidelines
- Be concise but insightful. Use bullet points and numbers.
- When discussing specific batches, reference their IDs (e.g., T001, T046).
- Provide actionable recommendations when relevant.
- Use domain terminology (pharmaceutical manufacturing).
- When asked about deviations, explain both the cause and the impact.
- Format numbers clearly (e.g., "82.5 kWh", "67.7 kg CO₂").
- Use emoji sparingly for key points (✅, ⚠️, 📊, 💡).
- Keep responses focused — avoid unnecessary padding.
"""
    return system_prompt


# ─── CONTEXT BUILDER ──────────────────────────────────────
def build_context_for_query(user_message):
    """
    Dynamically gather relevant batch data context based on
    keywords in the user's message.
    """
    df   = _load_df()
    sigs = _load_signatures()
    context_parts = []

    msg_lower = user_message.lower()

    # If user mentions a specific batch ID, include its full data
    for batch_id in df["Batch_ID"].tolist():
        if batch_id.lower() in msg_lower:
            batch = df[df["Batch_ID"] == batch_id].iloc[0]
            context_parts.append(
                f"### Batch {batch_id} Data:\n"
                f"{json.dumps(batch.to_dict(), indent=2, default=str)}"
            )

    # If asking about rankings/best/worst
    if any(kw in msg_lower for kw in ["rank", "best", "worst", "top", "bottom", "compare"]):
        top5 = df.nlargest(5, "Performance_Score")[["Batch_ID", "Quality_Score", "Yield_Score", "Total_Energy_kWh", "Carbon_kg_CO2", "Performance_Score"]]
        bottom5 = df.nsmallest(5, "Performance_Score")[["Batch_ID", "Quality_Score", "Yield_Score", "Total_Energy_kWh", "Carbon_kg_CO2", "Performance_Score"]]
        context_parts.append(
            f"### Top 5 Batches (by Performance):\n{top5.to_string(index=False)}\n\n"
            f"### Bottom 5 Batches (by Performance):\n{bottom5.to_string(index=False)}"
        )

    # If asking about energy/carbon/sustainability
    if any(kw in msg_lower for kw in ["energy", "carbon", "co2", "emission", "sustain", "green", "saving"]):
        golden_energy = sigs["balanced"]["benchmark_scores"]["Total_Energy_kWh"]
        golden_carbon = sigs["balanced"]["benchmark_scores"]["Carbon_kg_CO2"]
        avg_energy = df["Total_Energy_kWh"].mean()
        avg_carbon = df["Carbon_kg_CO2"].mean()
        context_parts.append(
            f"### Energy & Carbon Context:\n"
            f"- Fleet avg energy: {avg_energy:.2f} kWh/batch\n"
            f"- Golden target energy: {golden_energy:.2f} kWh/batch\n"
            f"- Potential energy savings per batch: {max(0, avg_energy - golden_energy):.2f} kWh\n"
            f"- Fleet avg carbon: {avg_carbon:.2f} kg CO₂/batch\n"
            f"- Golden target carbon: {golden_carbon:.2f} kg CO₂/batch\n"
            f"- Potential carbon savings per batch: {max(0, avg_carbon - golden_carbon):.2f} kg CO₂\n"
            f"- Total fleet energy: {df['Total_Energy_kWh'].sum():.2f} kWh\n"
            f"- Total fleet carbon: {df['Carbon_kg_CO2'].sum():.2f} kg CO₂"
        )

    # If asking about quality
    if any(kw in msg_lower for kw in ["quality", "hardness", "dissolution", "friability", "uniformity"]):
        context_parts.append(
            f"### Quality Context:\n"
            f"- Avg Quality Score: {df['Quality_Score'].mean():.2f}\n"
            f"- Quality range: {df['Quality_Score'].min():.2f} – {df['Quality_Score'].max():.2f}\n"
            f"- Std dev: {df['Quality_Score'].std():.2f}\n"
            f"- Golden quality target (balanced): {sigs['balanced']['benchmark_scores']['Quality_Score']}\n"
            f"- Golden quality target (quality mode): {sigs['quality']['benchmark_scores']['Quality_Score']}"
        )

    if context_parts:
        return "\n\n---\n\n## Relevant Data Context:\n\n" + "\n\n".join(context_parts)
    return ""


# ─── CHAT ─────────────────────────────────────────────────
def chat(user_message, conversation_history=None):
    """
    Send a message to Ollama (Llama 3.2) with full context.
    conversation_history: list of {"role": "user"/"assistant", "content": "..."}
    """
    system_prompt = build_system_prompt()

    # Build messages list for Ollama
    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add dynamic context to the user message
    context = build_context_for_query(user_message)
    enriched_message = user_message
    if context:
        enriched_message = f"{user_message}\n\n{context}"

    messages.append({"role": "user", "content": enriched_message})

    return _ollama_generate(messages, system=system_prompt, max_tokens=1024)


# ─── BATCH INSIGHT GENERATOR ─────────────────────────────
def generate_batch_insight(batch_id, mode="balanced"):
    """
    Auto-generate a natural language analysis of a batch's
    deviation from the golden signature.
    """
    # Import deviation engine here to avoid circular imports
    from deviation_engine import analyze_deviation

    report = analyze_deviation(batch_id, mode=mode)
    if "error" in report:
        return {"error": report["error"]}

    prompt = f"""Analyze this batch deviation report and provide a concise, actionable summary in 3-5 sentences.

## Deviation Report for Batch {batch_id} (Mode: {mode})

- Overall Status: {report['overall_status']}
- Critical Parameters: {report['summary']['critical_params']}
- Warning Parameters: {report['summary']['warning_params']}
- OK Parameters: {report['summary']['ok_params']}

### Parameter Deviations:
{json.dumps(report['param_analysis'], indent=2)}

### Outcome vs Golden Signature:
{json.dumps(report['outcome_comparison'], indent=2)}

### Savings Potential:
{json.dumps(report['savings_potential'], indent=2)}

Provide your analysis with:
1. A one-line status summary
2. The most critical issue (if any)
3. Top recommended action
4. Expected impact of fixing the issues
"""

    system = (
        "You are AuraOptima AI, an expert pharmaceutical manufacturing advisor. "
        "Provide concise, actionable batch analysis. Use bullet points. "
        "Be specific with numbers and batch IDs."
    )

    response_text = _ollama_generate(prompt, system=system, max_tokens=512)

    return {
        "batch_id": batch_id,
        "mode":     mode,
        "status":   report["overall_status"],
        "insight":  response_text,
        "summary":  report["summary"],
        "savings":  report["savings_potential"],
    }


# ─── MAIN (test) ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   AuraOptima — LLM Assistant Test (Ollama)")
    print("=" * 55)

    # Test chat
    print("\n📝 Testing chat...")
    response = chat("Which batch has the best quality score and why?")
    print(f"\n🤖 Response:\n{response}")

    # Test insight
    print("\n\n📝 Testing batch insight for T046...")
    insight = generate_batch_insight("T046")
    print(f"\n🤖 Insight:\n{insight['insight']}")
