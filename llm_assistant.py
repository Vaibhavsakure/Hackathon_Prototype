"""
AuraOptima — LLM Assistant Module
Uses Google Gemini API (primary) with local Ollama fallback
for natural language insights and conversational Q&A about
batch manufacturing data.
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

# Gemini config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Groq config (primary fallback — fast & free)
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"

# Ollama fallback config
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "gemma3:12b")

# Try to configure Gemini
_gemini_available = False
try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)  # type: ignore
        _gemini_available = True
        print(f"✅ Gemini API configured (model: {GEMINI_MODEL})")
    else:
        print("⚠️  GEMINI_API_KEY not set")
except ImportError:
    print("⚠️  google-generativeai not installed")

# Check Groq availability
_groq_available = bool(GROQ_API_KEY)
if _groq_available:
    print(f"✅ Groq API configured (model: {GROQ_MODEL})")
else:
    print("⚠️  GROQ_API_KEY not set")


# ─── LOAD HELPERS ─────────────────────────────────────────
def _load_signatures():
    with open(SIGNATURES_FILE) as f:
        return json.load(f)

def _load_df():
    return pd.read_csv(MASTER_CSV)


# ─── GEMINI API HELPER ───────────────────────────────────
def _gemini_generate(prompt, system=None, max_tokens=1024):
    """Call Google Gemini API for text generation."""
    import google.generativeai as genai

    model = genai.GenerativeModel(  # type: ignore
        model_name=GEMINI_MODEL,
        system_instruction=system,
        generation_config=genai.types.GenerationConfig(  # type: ignore
            max_output_tokens=max_tokens,
            temperature=0.7,
        ),
    )

    # Build conversation from messages list or plain string
    if isinstance(prompt, list):
        # Convert chat history format to Gemini format
        history = []
        last_user_msg = ""
        for msg in prompt:
            role = "user" if msg["role"] == "user" else "model"
            if msg == prompt[-1] and role == "user":
                last_user_msg = msg["content"]
            else:
                history.append({"role": role, "parts": [msg["content"]]})

        chat = model.start_chat(history=history)
        response = chat.send_message(last_user_msg)
    else:
        response = model.generate_content(prompt)

    return response.text


# ─── GROQ API HELPER (PRIMARY FALLBACK) ──────────────────
def _groq_generate(prompt, system=None, max_tokens=1024):
    """Call Groq API via REST (OpenAI-compatible endpoint)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if isinstance(prompt, list):
        for msg in prompt:
            role = msg["role"]
            if role == "model":
                role = "assistant"
            messages.append({"role": role, "content": msg["content"]})
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(
            GROQ_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.HTTPError as e:
        raise RuntimeError(f"Groq API error: {e.response.status_code} — {e.response.text[:200]}")
    except Exception as e:
        raise RuntimeError(f"Groq error: {str(e)}")


# ─── OLLAMA API HELPER (LAST FALLBACK) ───────────────────
def _ollama_generate(prompt, system=None, max_tokens=1024):
    """Call Ollama's local chat API as last-resort fallback."""
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


# ─── SMART GENERATE (Gemini → Groq → Ollama) ────────────
def _generate(prompt, system=None, max_tokens=1024):
    """
    Try Gemini API first. If it fails, try Groq.
    If Groq fails, fall back to local Ollama.
    """
    errors = []

    # 1. Try Gemini first
    if _gemini_available:
        try:
            return _gemini_generate(prompt, system=system, max_tokens=max_tokens)
        except Exception as e:
            errors.append(f"Gemini: {str(e)[:100]}")
            print(f"⚠️  Gemini API failed: {str(e)[:100]}")
            print(f"   Falling back to Groq...")

    # 2. Try Groq
    if _groq_available:
        try:
            return _groq_generate(prompt, system=system, max_tokens=max_tokens)
        except Exception as e:
            errors.append(f"Groq: {str(e)[:100]}")
            print(f"⚠️  Groq API failed: {str(e)[:100]}")
            print(f"   Falling back to Ollama ({OLLAMA_MODEL})...")

    # 3. Last resort: Ollama
    try:
        return _ollama_generate(prompt, system=system, max_tokens=max_tokens)
    except Exception as e:
        errors.append(f"Ollama: {str(e)[:100]}")
        raise RuntimeError(
            f"All LLM backends failed.\n"
            + "\n".join(errors) + "\n"
            f"Set GROQ_API_KEY in .env (free at https://console.groq.com)"
        )


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
- When asked "why" questions, perform a why-why root cause analysis:
  1. What happened? (status and key metric)
  2. Which parameters caused it? (top deviating parameters with contribution %)
  3. Why did those parameters deviate? (relate to process conditions)
  4. What is the fix? (specific actionable recommendation with numbers)
  Chain these together so the operator understands the full cause-effect path.
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
    Send a message with full context.
    Tries Gemini API first, falls back to local Ollama if needed.
    conversation_history: list of {"role": "user"/"assistant", "content": "..."}
    """
    system_prompt = build_system_prompt()

    # Build messages list
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

    return _generate(messages, system=system_prompt, max_tokens=1024)


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
        return report
    summary = report["summary"] if isinstance(report.get("summary"), dict) else {}

    prompt = f"""Analyze this batch deviation report and provide a concise, actionable summary in 3-5 sentences.

## Deviation Report for Batch {batch_id} (Mode: {mode})

- Overall Status: {report.get('overall_status')}
- Critical Parameters: {summary.get('critical_params')}
- Warning Parameters: {summary.get('warning_params')}
- OK Parameters: {summary.get('ok_params')}

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

    response_text = _generate(prompt, system=system, max_tokens=512)

    return {
        "batch_id": batch_id,
        "mode":     mode,
        "status":   report["overall_status"],
        "insight":  response_text,
        "summary":  report["summary"],
        "savings":  report["savings_potential"],
    }


# ─── WHY-WHY ROOT CAUSE ANALYSIS ─────────────────────────
def generate_why_why_analysis(batch_id, mode="balanced"):
    """
    Generate a structured why-why root cause analysis for a batch.
    This chains together: What happened → Why → Why deeper → Fix.
    """
    from deviation_engine import analyze_deviation

    report = analyze_deviation(batch_id, mode=mode)
    if "error" in report:
        return report

    summary = report["summary"] if isinstance(report.get("summary"), dict) else {}
    param_analysis = report.get("param_analysis", {})

    # Build the deviating parameters list
    deviating = []
    for param, info in param_analysis.items():
        if info["severity"] != "OK":
            deviating.append({
                "param": param,
                "deviation": info["deviation_pct"],
                "severity": info["severity"],
                "direction": info["direction"],
                "batch_val": info["batch_value"],
                "golden_val": info["golden_value"],
            })
    deviating.sort(key=lambda x: abs(x["deviation"]), reverse=True)

    prompt = f"""Perform a structured WHY-WHY root cause analysis for Batch {batch_id}.

Use this exact format:

## Level 1: WHAT happened?
Describe the batch status and key metrics.

## Level 2: WHICH parameters caused it?
List the top deviating parameters with their contribution.

## Level 3: WHY did those parameters deviate?
Explain the likely process conditions that led to each deviation.

## Level 4: ROOT CAUSE
Identify the most likely root cause.

## Level 5: RECOMMENDED FIX
Provide specific actionable steps with numbers.

## Deviation Data:
- Overall Status: {report.get('overall_status')}
- Critical Parameters: {summary.get('critical_params', 0)}
- Warning Parameters: {summary.get('warning_params', 0)}

### Deviating Parameters:
{json.dumps(deviating, indent=2)}

### Outcome vs Golden:
{json.dumps(report['outcome_comparison'], indent=2)}

### Savings if fixed:
{json.dumps(report['savings_potential'], indent=2)}

Be specific with batch IDs, parameter names, and numbers.
Keep each level concise (2-3 sentences max).
"""

    system = (
        "You are AuraOptima AI. Perform structured root cause analysis. "
        "Use the exact Level 1-5 format requested. Be specific and actionable."
    )

    analysis_text = _generate(prompt, system=system, max_tokens=800)

    return {
        "batch_id": batch_id,
        "mode": mode,
        "status": report["overall_status"],
        "why_why_analysis": analysis_text,
        "deviating_parameters": deviating,
        "summary": report["summary"],
        "savings": report["savings_potential"],
    }


# ─── MAIN (test) ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   AuraOptima — LLM Assistant Test")
    print("   Gemini API → Ollama Fallback")
    print("=" * 55)

    # Test chat
    print("\n📝 Testing chat...")
    response = chat("Which batch has the best quality score and why?")
    print(f"\n🤖 Response:\n{response}")

    # Test insight
    print("\n\n📝 Testing batch insight for T046...")
    insight = generate_batch_insight("T046")
    print(f"\n🤖 Insight:\n{insight['insight']}")
