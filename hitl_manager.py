import json
import os
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────
SIGNATURES_FILE  = "data/golden_signatures.json"
DECISIONS_FILE   = "data/decisions_log.json"
PROPOSALS_FILE   = "data/pending_proposals.json"

# ─── LOAD / INIT ──────────────────────────────────────────
def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

def save_json(path, data):
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ─── PROPOSE SIGNATURE UPDATE ─────────────────────────────
def propose_update(mode, batch_id, new_params, new_scores, improvement_pct):
    """Create a pending proposal for human review."""
    proposals = load_json(PROPOSALS_FILE, {})
    proposal_id = f"PROP_{mode.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    proposals[proposal_id] = {
        "proposal_id":    proposal_id,
        "mode":           mode,
        "batch_id":       batch_id,
        "new_params":     new_params,
        "new_scores":     new_scores,
        "improvement_pct": improvement_pct,
        "status":         "PENDING",
        "created_at":     datetime.now().isoformat(),
        "reviewed_at":    None,
        "reviewer_note":  None,
    }

    save_json(PROPOSALS_FILE, proposals)
    print(f"📋 Proposal created: {proposal_id}")
    return proposal_id

# ─── ACCEPT PROPOSAL ──────────────────────────────────────
def accept_proposal(proposal_id, reviewer_note=""):
    """Accept a proposal → update the golden signature."""
    proposals  = load_json(PROPOSALS_FILE, {})
    signatures = load_json(SIGNATURES_FILE, {})
    decisions  = load_json(DECISIONS_FILE, [])

    if proposal_id not in proposals:
        print(f"❌ Proposal {proposal_id} not found")
        return False

    prop = proposals[proposal_id]
    if prop["status"] != "PENDING":
        print(f"⚠️  Proposal {proposal_id} already {prop['status']}")
        return False

    mode = prop["mode"]

    # ── Update golden signature ──
    old_version = signatures[mode].get("version", 1)
    signatures[mode]["optimal_params"]     = prop["new_params"]
    signatures[mode]["benchmark_scores"]   = prop["new_scores"]
    signatures[mode]["source_batch"]       = prop["batch_id"]
    signatures[mode]["version"]            = old_version + 1
    signatures[mode]["last_updated"]       = datetime.now().isoformat()

    # ── Update proposal status ──
    prop["status"]      = "ACCEPTED"
    prop["reviewed_at"] = datetime.now().isoformat()
    prop["reviewer_note"] = reviewer_note

    # ── Log decision ──
    decisions.append({
        "proposal_id":   proposal_id,
        "action":        "ACCEPTED",
        "mode":          mode,
        "batch_id":      prop["batch_id"],
        "improvement":   prop["improvement_pct"],
        "reviewer_note": reviewer_note,
        "timestamp":     datetime.now().isoformat(),
    })

    save_json(SIGNATURES_FILE, signatures)
    save_json(PROPOSALS_FILE,  proposals)
    save_json(DECISIONS_FILE,  decisions)

    print(f"✅ Proposal {proposal_id} ACCEPTED → '{mode}' signature updated to v{old_version + 1}")
    return True

# ─── REJECT PROPOSAL ──────────────────────────────────────
def reject_proposal(proposal_id, reviewer_note=""):
    """Reject a proposal → keep existing signature."""
    proposals = load_json(PROPOSALS_FILE, {})
    decisions = load_json(DECISIONS_FILE, [])

    if proposal_id not in proposals:
        print(f"❌ Proposal {proposal_id} not found")
        return False

    prop = proposals[proposal_id]
    prop["status"]        = "REJECTED"
    prop["reviewed_at"]   = datetime.now().isoformat()
    prop["reviewer_note"] = reviewer_note

    decisions.append({
        "proposal_id":   proposal_id,
        "action":        "REJECTED",
        "mode":          prop["mode"],
        "batch_id":      prop["batch_id"],
        "improvement":   prop["improvement_pct"],
        "reviewer_note": reviewer_note,
        "timestamp":     datetime.now().isoformat(),
    })

    save_json(PROPOSALS_FILE, proposals)
    save_json(DECISIONS_FILE, decisions)

    print(f"❌ Proposal {proposal_id} REJECTED — signature unchanged")
    return True

# ─── REPRIORITIZE MODE ────────────────────────────────────
def reprioritize_mode(mode, new_weights, reason=""):
    """Change weights for a signature mode."""
    signatures = load_json(SIGNATURES_FILE, {})
    decisions  = load_json(DECISIONS_FILE, [])

    if mode not in signatures:
        print(f"❌ Mode '{mode}' not found")
        return False

    old_weights = signatures[mode]["weights"]
    signatures[mode]["weights"]      = new_weights
    signatures[mode]["last_updated"] = datetime.now().isoformat()

    decisions.append({
        "action":      "REPRIORITIZED",
        "mode":        mode,
        "old_weights": old_weights,
        "new_weights": new_weights,
        "reason":      reason,
        "timestamp":   datetime.now().isoformat(),
    })

    save_json(SIGNATURES_FILE, signatures)
    save_json(DECISIONS_FILE,  decisions)

    print(f"🔄 Mode '{mode}' reprioritized")
    print(f"   Old: {old_weights}")
    print(f"   New: {new_weights}")
    return True

# ─── GET PENDING PROPOSALS ────────────────────────────────
def get_pending_proposals():
    proposals = load_json(PROPOSALS_FILE, {})
    return {k: v for k, v in proposals.items() if v["status"] == "PENDING"}

# ─── DECISIONS SUMMARY ────────────────────────────────────
def get_decisions_summary():
    decisions = load_json(DECISIONS_FILE, [])
    accepted  = sum(1 for d in decisions if d.get("action") == "ACCEPTED")
    rejected  = sum(1 for d in decisions if d.get("action") == "REJECTED")
    reprioritized = sum(1 for d in decisions if d.get("action") == "REPRIORITIZED")
    return {
        "total":         len(decisions),
        "accepted":      accepted,
        "rejected":      rejected,
        "reprioritized": reprioritized,
        "history":       decisions
    }

# ─── MAIN DEMO ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   AuraOptima — HITL Manager")
    print("=" * 55)

    # Simulate a proposal (T046 beats 'quality' mode by 4.53%)
    signatures = load_json(SIGNATURES_FILE, {})
    q_sig = signatures["quality"]

    prop_id = propose_update(
        mode           = "quality",
        batch_id       = "T046",
        new_params     = {
            "Granulation_Time": 12.0, "Binder_Amount": 7.3,
            "Drying_Temp": 66.0,      "Drying_Time": 20.0,
            "Compression_Force": 15.2, "Machine_Speed": 122.0,
            "Lubricant_Conc": 0.6
        },
        new_scores     = {
            "Quality_Score": 83.182, "Yield_Score": 50.521,
            "Total_Energy_kWh": 75.724, "Carbon_kg_CO2": 62.094,
            "Performance_Score": 75.45, "composite_score": 0.7856
        },
        improvement_pct = 4.53
    )

    print(f"\n📋 Pending Proposals: {len(get_pending_proposals())}")

    # Simulate engineer ACCEPTS the proposal
    print("\n🧑‍🔬 Engineer Decision: ACCEPT")
    accept_proposal(prop_id, reviewer_note="T046 shows better balanced performance. Approved.")

    # Simulate reprioritization
    print("\n🔄 Reprioritizing 'energy' mode (shift focus more to carbon):")
    reprioritize_mode(
        mode        = "energy",
        new_weights = {"quality": 0.1, "yield": 0.15, "energy": 0.75},
        reason      = "Sustainability target: reduce carbon by 15% this quarter"
    )

    # Summary
    summary = get_decisions_summary()
    print(f"\n📊 Decisions Summary:")
    print(f"   Total   : {summary['total']}")
    print(f"   Accepted: {summary['accepted']}")
    print(f"   Rejected: {summary['rejected']}")
    print(f"   Reprioritized: {summary['reprioritized']}")

    print("\n🎉 HITL Manager complete!")
    print(f"   → data/decisions_log.json")
    print(f"   → data/pending_proposals.json")