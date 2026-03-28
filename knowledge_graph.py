"""
AuraOptima — Manufacturing Knowledge Graph
══════════════════════════════════════════════════════════
Builds a living knowledge graph connecting all manufacturing
entities for multi-hop reasoning (Track A requirement).

Relationships:
  Asset → Produces → Batch
  Batch → Has → Energy Patterns
  Energy Patterns → Indicates → Asset Health Events
  Process Parameters → Influences → Energy Patterns
  Batch → Compared Against → Golden Signature
  Golden Signature → Optimized For → Objectives
  Raw Material → Affects → Yield Outcome
  Anomaly → Triggered By → Process Drift
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

MASTER_CSV      = "data/master_df.csv"
SIGNATURES_FILE = "data/golden_signatures.json"

# ─── THRESHOLDS ───────────────────────────────────────────
ANOMALY_THRESHOLD = 20.0  # % deviation from golden = anomaly
HIGH_ENERGY_THRESHOLD = 90.0  # kWh — high energy pattern
LOW_YIELD_THRESHOLD = 45.0  #yield score — low yield


# ═══════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH BUILDER
# ═══════════════════════════════════════════════════════════
def build_graph():
    """
    Build the full knowledge graph from batch data and golden signatures.
    Returns {nodes: [...], edges: [...]} structure.
    """
    df = pd.read_csv(MASTER_CSV)
    with open(SIGNATURES_FILE) as f:
        signatures = json.load(f)

    nodes = []
    edges = []
    node_ids = set()

    def add_node(nid, ntype, label, data=None):
        if nid not in node_ids:
            node_ids.add(nid)
            nodes.append({
                "id": nid,
                "type": ntype,
                "label": label,
                "data": data or {},
            })

    def add_edge(source, target, relationship, data=None):
        edges.append({
            "source": source,
            "target": target,
            "relationship": relationship,
            "data": data or {},
        })

    # ─── 1. ASSET nodes (production lines) ────────────────
    add_node("ASSET_LINE1", "asset", "Production Line 1", {
        "type": "Pharmaceutical Tablet Press",
        "capacity": "60 batches",
    })

    # ─── 2. BATCH nodes ──────────────────────────────────
    for _, row in df.iterrows():
        bid = row["Batch_ID"]
        add_node(f"BATCH_{bid}", "batch", bid, {
            "quality_score": round(float(row["Quality_Score"]), 2),
            "yield_score": round(float(row["Yield_Score"]), 2),
            "energy_kWh": round(float(row["Total_Energy_kWh"]), 2),
            "carbon_kg": round(float(row["Carbon_kg_CO2"]), 2),
            "performance": round(float(row["Performance_Score"]), 2),
        })

        # Edge: Asset → Produces → Batch
        add_edge("ASSET_LINE1", f"BATCH_{bid}", "produces")

    # ─── 3. ENERGY PATTERN nodes ─────────────────────────
    for _, row in df.iterrows():
        bid = row["Batch_ID"]
        energy = float(row["Total_Energy_kWh"])

        if energy > HIGH_ENERGY_THRESHOLD:
            pattern = "HIGH_ENERGY"
            pattern_label = f"High Energy ({energy:.0f} kWh)"
        elif energy < HIGH_ENERGY_THRESHOLD * 0.7:
            pattern = "LOW_ENERGY"
            pattern_label = f"Low Energy ({energy:.0f} kWh)"
        else:
            pattern = "NORMAL_ENERGY"
            pattern_label = f"Normal Energy ({energy:.0f} kWh)"

        ep_id = f"EP_{bid}"
        add_node(ep_id, "energy_pattern", pattern_label, {
            "pattern": pattern,
            "total_energy_kWh": round(energy, 2),
            "carbon_kg": round(float(row["Carbon_kg_CO2"]), 2),
            "batch_id": bid,
        })

        # Edge: Batch → Has → Energy Pattern
        add_edge(f"BATCH_{bid}", ep_id, "has_energy_pattern")

    # ─── 4. ASSET HEALTH EVENTS ──────────────────────────
    high_energy_batches = df[df["Total_Energy_kWh"] > HIGH_ENERGY_THRESHOLD]
    if len(high_energy_batches) > 0:
        add_node("HEALTH_HIGH_CONSUMPTION", "health_event", "High Energy Consumption Alert", {
            "affected_batches": len(high_energy_batches),
            "avg_excess_kWh": round(float(high_energy_batches["Total_Energy_kWh"].mean() - HIGH_ENERGY_THRESHOLD), 2),
        })
        for _, row in high_energy_batches.iterrows():
            # Edge: Energy Pattern → Indicates → Asset Health Event
            add_edge(f"EP_{row['Batch_ID']}", "HEALTH_HIGH_CONSUMPTION", "indicates")

    high_vibration = df[df["Avg_Vibration"] > df["Avg_Vibration"].quantile(0.9)]
    if len(high_vibration) > 0:
        add_node("HEALTH_VIBRATION_ALERT", "health_event", "High Vibration Alert", {
            "affected_batches": len(high_vibration),
        })
        for _, row in high_vibration.iterrows():
            add_edge(f"EP_{row['Batch_ID']}", "HEALTH_VIBRATION_ALERT", "indicates")

    # ─── 5. PROCESS PARAMETERS ───────────────────────────
    decision_vars = [
        "Granulation_Time", "Binder_Amount", "Drying_Temp",
        "Drying_Time", "Compression_Force", "Machine_Speed", "Lubricant_Conc"
    ]

    for var in decision_vars:
        add_node(f"PARAM_{var}", "process_parameter", var.replace("_", " "), {
            "min": round(float(df[var].min()), 2),
            "max": round(float(df[var].max()), 2),
            "mean": round(float(df[var].mean()), 2),
            "std": round(float(df[var].std()), 2),
        })

        # Edge: Process Parameter → Influences → Energy Patterns
        # Calculate correlation between parameter and energy
        corr = float(df[var].corr(df["Total_Energy_kWh"]))
        if abs(corr) > 0.2:
            influence = "strong" if abs(corr) > 0.5 else "moderate"
            for _, row in df.iterrows():
                add_edge(f"PARAM_{var}", f"EP_{row['Batch_ID']}", "influences", {
                    "correlation": round(corr, 3),
                    "strength": influence,
                })
                break  # Only add one representative edge per parameter

    # ─── 6. GOLDEN SIGNATURES ────────────────────────────
    for mode, sig in signatures.items():
        gs_id = f"GS_{mode}"
        add_node(gs_id, "golden_signature", f"Golden Signature ({mode.title()})", {
            "mode": mode,
            "label": sig["label"],
            "version": sig.get("version", 1),
            "source_batch": sig["source_batch"],
            "benchmark": sig["benchmark_scores"],
            "weights": sig["weights"],
        })

        # Edge: Batch → Compared Against → Golden Signature
        for _, row in df.iterrows():
            add_edge(f"BATCH_{row['Batch_ID']}", gs_id, "compared_against")

        # Edge: Golden Signature → Optimized For → Objectives
        objectives = list(sig["weights"].keys())
        for obj in objectives:
            obj_id = f"OBJ_{obj}"
            add_node(obj_id, "objective", obj.title(), {
                "weight_in_" + mode: sig["weights"][obj],
            })
            add_edge(gs_id, obj_id, "optimized_for", {"weight": sig["weights"][obj]})

    # ─── 7. RAW MATERIAL → YIELD ─────────────────────────
    add_node("RAW_BINDER", "raw_material", "Binder Material", {
        "type": "Granulation Binder",
        "parameter": "Binder_Amount",
    })
    add_node("RAW_LUBRICANT", "raw_material", "Lubricant", {
        "type": "Tablet Lubricant",
        "parameter": "Lubricant_Conc",
    })

    for _, row in df.iterrows():
        bid = row["Batch_ID"]
        # Edge: Raw Material → Affects → Yield
        add_edge("RAW_BINDER", f"BATCH_{bid}", "affects_yield", {
            "binder_amount": round(float(row["Binder_Amount"]), 2),
            "yield_score": round(float(row["Yield_Score"]), 2),
        })
        add_edge("RAW_LUBRICANT", f"BATCH_{bid}", "affects_yield", {
            "lubricant_conc": round(float(row["Lubricant_Conc"]), 3),
            "yield_score": round(float(row["Yield_Score"]), 2),
        })

    # ─── 8. ANOMALY → PROCESS DRIFT ─────────────────────
    # Detect anomalies: batches with >20% avg deviation from golden
    golden = signatures.get("balanced", {}).get("optimal_params", {})
    for _, row in df.iterrows():
        bid = row["Batch_ID"]
        total_dev = 0
        for var in decision_vars:
            gval = float(golden.get(var, 0))
            bval = float(row[var])
            if gval != 0:
                total_dev += abs((bval - gval) / gval * 100)
        avg_dev = total_dev / len(decision_vars)

        if avg_dev >= ANOMALY_THRESHOLD:
            anom_id = f"ANOMALY_{bid}"
            add_node(anom_id, "anomaly", f"Process Drift ({bid})", {
                "batch_id": bid,
                "avg_deviation_pct": round(avg_dev, 1),
                "severity": "CRITICAL" if avg_dev > 30 else "WARNING",
            })

            # Edge: Anomaly → Triggered By → Process Drift
            add_edge(anom_id, f"BATCH_{bid}", "triggered_by_drift")

            # Find which params drifted most
            for var in decision_vars:
                gval = float(golden.get(var, 0))
                bval = float(row[var])
                if gval != 0:
                    dev = abs((bval - gval) / gval * 100)
                    if dev >= ANOMALY_THRESHOLD:
                        add_edge(anom_id, f"PARAM_{var}", "caused_by", {
                            "deviation_pct": round(dev, 1),
                        })

    return {"nodes": nodes, "edges": edges}


# ═══════════════════════════════════════════════════════════
# QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════
def get_node_relationships(graph, node_id):
    """Get all relationships for a specific node (multi-hop ready)."""
    connected = []
    for edge in graph["edges"]:
        if edge["source"] == node_id:
            target_node = next((n for n in graph["nodes"] if n["id"] == edge["target"]), None)
            connected.append({
                "direction": "outgoing",
                "relationship": edge["relationship"],
                "node": target_node,
                "edge_data": edge.get("data", {}),
            })
        elif edge["target"] == node_id:
            source_node = next((n for n in graph["nodes"] if n["id"] == edge["source"]), None)
            connected.append({
                "direction": "incoming",
                "relationship": edge["relationship"],
                "node": source_node,
                "edge_data": edge.get("data", {}),
            })
    return connected


def get_graph_summary():
    """Get a summary of the graph for display."""
    graph = build_graph()
    type_counts = {}
    for node in graph["nodes"]:
        t = node["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    rel_counts = {}
    for edge in graph["edges"]:
        r = edge["relationship"]
        rel_counts[r] = rel_counts.get(r, 0) + 1

    return {
        "total_nodes": len(graph["nodes"]),
        "total_edges": len(graph["edges"]),
        "node_types": type_counts,
        "relationship_types": rel_counts,
    }


def get_visualization_graph():
    """
    Get a simplified graph for frontend visualization.
    Groups edges and limits to key relationships to avoid visual clutter.
    """
    graph = build_graph()

    # For visualization, we reduce to representative nodes
    viz_nodes = []
    viz_edges = []
    seen_edge_types = set()

    # Include all non-batch nodes
    for node in graph["nodes"]:
        if node["type"] != "batch" or node["id"] in ["BATCH_T001", "BATCH_T038", "BATCH_T046"]:
            viz_nodes.append({
                "id": node["id"],
                "type": node["type"],
                "label": node["label"],
                "data": node.get("data", {}),
            })

    viz_node_ids = {n["id"] for n in viz_nodes}

    # Include edges connecting visualization nodes
    for edge in graph["edges"]:
        if edge["source"] in viz_node_ids and edge["target"] in viz_node_ids:
            edge_key = f"{edge['source']}-{edge['relationship']}-{edge['target']}"
            if edge_key not in seen_edge_types:
                seen_edge_types.add(edge_key)
                viz_edges.append(edge)

    return {"nodes": viz_nodes, "edges": viz_edges}


def query_path(from_node, to_node, max_depth=5):
    """
    BFS multi-hop traversal between two nodes in the knowledge graph.
    Returns the shortest path as a list of nodes and edges.
    """
    graph = build_graph()
    # Build adjacency list
    adjacency = {}
    for edge in graph["edges"]:
        adjacency.setdefault(edge["source"], []).append(edge)
        # bidirectional for traversal
        adjacency.setdefault(edge["target"], []).append({
            "source": edge["target"],
            "target": edge["source"],
            "relationship": edge["relationship"],
            "data": edge.get("data", {}),
        })

    # BFS
    from collections import deque
    queue = deque([(from_node, [from_node], [])])
    visited = {from_node}

    while queue:
        current, path_nodes, path_edges = queue.popleft()
        if current == to_node:
            # Resolve node objects
            node_map = {n["id"]: n for n in graph["nodes"]}
            return {
                "found": True,
                "from": from_node,
                "to": to_node,
                "hops": len(path_edges),
                "path_nodes": [node_map.get(nid, {"id": nid}) for nid in path_nodes],
                "path_edges": path_edges,
            }
        if len(path_nodes) > max_depth:
            continue
        for edge in adjacency.get(current, []):
            neighbor = edge["target"]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((
                    neighbor,
                    path_nodes + [neighbor],
                    path_edges + [{"from": current, "to": neighbor, "relationship": edge["relationship"]}],
                ))

    return {"found": False, "from": from_node, "to": to_node, "hops": 0, "path_nodes": [], "path_edges": []}


def get_natural_language_summary(node_id):
    """
    Use LLM to generate a plain-English summary of a node
    and all its relationships in the knowledge graph.
    """
    graph = build_graph()
    node = next((n for n in graph["nodes"] if n["id"] == node_id), None)
    if not node:
        return {"error": f"Node {node_id} not found"}

    rels = get_node_relationships(graph, node_id)

    # Build context for LLM
    rel_descriptions = []
    for r in rels[:20]:  # limit to 20 for prompt size
        rn = r.get("node")
        if rn:
            direction = "→" if r["direction"] == "outgoing" else "←"
            rel_descriptions.append(
                f"  {direction} [{r['relationship']}] {rn['label']} ({rn['type']})"
            )

    prompt = f"""Summarize this manufacturing knowledge graph node and its relationships in 3-4 concise sentences.

Node: {node['label']} (Type: {node['type']})
Data: {json.dumps(node.get('data', {}), indent=2)}

Relationships ({len(rels)} total):
{chr(10).join(rel_descriptions)}

Be specific with numbers. Mention the node type, key metrics, and how it connects to other entities."""

    try:
        from llm_assistant import _generate
        summary_text = _generate(prompt, system="You are AuraOptima AI. Provide concise manufacturing knowledge graph summaries.", max_tokens=300)
    except Exception:
        # Fallback: generate a simple summary without LLM
        summary_text = (
            f"{node['label']} is a {node['type']} node with {len(rels)} relationships. "
        )
        data = node.get("data", {})
        if data:
            details = ", ".join(f"{k}: {v}" for k, v in list(data.items())[:4])
            summary_text += f"Key data: {details}."

    return {
        "node_id": node_id,
        "label": node["label"],
        "type": node["type"],
        "summary": summary_text,
        "relationship_count": len(rels),
        "data": node.get("data", {}),
    }


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   AuraOptima — Knowledge Graph Builder")
    print("=" * 55)

    graph = build_graph()
    summary = get_graph_summary()

    print(f"\n📊 Graph Summary:")
    print(f"   Total Nodes: {summary['total_nodes']}")
    print(f"   Total Edges: {summary['total_edges']}")
    print(f"\n   Node Types:")
    for ntype, count in summary["node_types"].items():
        print(f"      {ntype}: {count}")
    print(f"\n   Relationship Types:")
    for rel, count in summary["relationship_types"].items():
        print(f"      {rel}: {count}")

    # Demo: query relationships for a batch
    print(f"\n\n🔍 Relationships for Batch T038:")
    rels = get_node_relationships(graph, "BATCH_T038")
    for r in rels[:10]:
        node = r["node"]
        print(f"   {'→' if r['direction'] == 'outgoing' else '←'} "
              f"[{r['relationship']}] {node['label'] if node else '?'} ({node['type'] if node else '?'})")

    print("\n🎉 Knowledge Graph complete!")
