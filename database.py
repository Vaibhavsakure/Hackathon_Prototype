"""
AuraOptima — SQLite Database Layer
══════════════════════════════════════════════════════════
Provides persistent storage for batch data, HITL decisions,
and signature proposals using SQLite.
Auto-seeds from CSV/JSON files on first run.
"""

import sqlite3
import json
import os
import csv
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "data/auraoptima.db"
MASTER_CSV = "data/master_df.csv"
DECISIONS_JSON = "data/decisions_log.json"
PROPOSALS_JSON = "data/pending_proposals.json"

# ─── CONNECTION MANAGER ───────────────────────────────────
@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ─── SCHEMA ───────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    Batch_ID TEXT PRIMARY KEY,
    Granulation_Time REAL,
    Binder_Amount REAL,
    Drying_Temp REAL,
    Drying_Time REAL,
    Compression_Force REAL,
    Machine_Speed REAL,
    Lubricant_Conc REAL,
    Moisture_Content REAL,
    Tablet_Weight REAL,
    Hardness REAL,
    Friability REAL,
    Disintegration_Time REAL,
    Dissolution_Rate REAL,
    Content_Uniformity REAL,
    Total_Energy_kWh REAL,
    Avg_Power_kW REAL,
    Max_Power_kW REAL,
    Avg_Temperature REAL,
    Max_Temperature REAL,
    Avg_Vibration REAL,
    Max_Vibration REAL,
    Avg_Motor_Speed REAL,
    Batch_Duration_min REAL,
    Energy_Blending_kWh REAL,
    Energy_Coating_kWh REAL,
    Energy_Compression_kWh REAL,
    Energy_Drying_kWh REAL,
    Energy_Granulation_kWh REAL,
    Energy_Milling_kWh REAL,
    Energy_Preparation_kWh REAL,
    Energy_Quality_Testing_kWh REAL,
    Carbon_kg_CO2 REAL,
    Quality_Score REAL,
    Yield_Score REAL,
    Energy_Efficiency REAL,
    Performance_Score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT,
    action TEXT NOT NULL,
    mode TEXT,
    batch_id TEXT,
    improvement REAL,
    reviewer_note TEXT,
    old_weights TEXT,
    new_weights TEXT,
    reason TEXT,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    new_params TEXT NOT NULL,
    new_scores TEXT NOT NULL,
    improvement_pct REAL,
    status TEXT DEFAULT 'PENDING',
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    reviewer_note TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_action ON decisions(action);
CREATE INDEX IF NOT EXISTS idx_decisions_mode ON decisions(mode);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status);
CREATE INDEX IF NOT EXISTS idx_batches_quality ON batches(Quality_Score);
CREATE INDEX IF NOT EXISTS idx_batches_energy ON batches(Total_Energy_kWh);
"""

# ─── INITIALIZATION ──────────────────────────────────────
def init_db():
    """Create tables and seed data from CSV/JSON if tables are empty."""
    os.makedirs("data", exist_ok=True)
    
    with get_db() as conn:
        conn.executescript(SCHEMA)
        
        # Check if batches table is empty
        count = conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        if count == 0:
            _seed_batches(conn)
            print(f"✅ Database seeded with batch data from {MASTER_CSV}")
        
        # Check if decisions table is empty
        count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        if count == 0:
            _seed_decisions(conn)
            print(f"✅ Database seeded with HITL decisions from {DECISIONS_JSON}")
        
        # Check if proposals table is empty
        count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        if count == 0:
            _seed_proposals(conn)
            print(f"✅ Database seeded with proposals from {PROPOSALS_JSON}")
    
    print(f"🗄️  Database ready: {DB_PATH}")

def _seed_batches(conn):
    """Load batch data from CSV into SQLite."""
    if not os.path.exists(MASTER_CSV):
        print(f"⚠️  {MASTER_CSV} not found, skipping batch seed")
        return
    
    with open(MASTER_CSV, 'r') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        if not cols:
            return
        placeholders = ','.join(['?' for _ in cols])
        col_names = ','.join(cols)
        
        for row in reader:
            values = []
            for col in cols:
                val = row[col]
                # Try to convert to float for numeric columns
                if col != 'Batch_ID':
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
                values.append(val)
            
            conn.execute(
                f"INSERT OR IGNORE INTO batches ({col_names}) VALUES ({placeholders})",
                values
            )

def _seed_decisions(conn):
    """Load HITL decisions from JSON into SQLite."""
    if not os.path.exists(DECISIONS_JSON):
        return
    
    with open(DECISIONS_JSON) as f:
        decisions = json.load(f)
    
    for d in decisions:
        conn.execute(
            """INSERT INTO decisions (proposal_id, action, mode, batch_id, 
               improvement, reviewer_note, old_weights, new_weights, reason, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                d.get("proposal_id"),
                d.get("action"),
                d.get("mode"),
                d.get("batch_id"),
                d.get("improvement"),
                d.get("reviewer_note"),
                json.dumps(d.get("old_weights")) if d.get("old_weights") else None,
                json.dumps(d.get("new_weights")) if d.get("new_weights") else None,
                d.get("reason"),
                d.get("timestamp", datetime.now().isoformat()),
            )
        )

def _seed_proposals(conn):
    """Load proposals from JSON into SQLite."""
    if not os.path.exists(PROPOSALS_JSON):
        return
    
    with open(PROPOSALS_JSON) as f:
        proposals = json.load(f)
    
    for pid, p in proposals.items():
        conn.execute(
            """INSERT OR IGNORE INTO proposals 
               (proposal_id, mode, batch_id, new_params, new_scores, 
                improvement_pct, status, created_at, reviewed_at, reviewer_note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pid,
                p.get("mode"),
                p.get("batch_id"),
                json.dumps(p.get("new_params", {})),
                json.dumps(p.get("new_scores", {})),
                p.get("improvement_pct"),
                p.get("status", "PENDING"),
                p.get("created_at", datetime.now().isoformat()),
                p.get("reviewed_at"),
                p.get("reviewer_note"),
            )
        )

# ═══════════════════════════════════════════════════════════
# BATCH CRUD OPERATIONS
# ═══════════════════════════════════════════════════════════

def get_all_batches():
    """Retrieve all batches as list of dicts."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM batches ORDER BY Batch_ID").fetchall()
        return [dict(row) for row in rows]

def get_batch(batch_id):
    """Retrieve a single batch by ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM batches WHERE Batch_ID = ?", (batch_id,)).fetchone()
        return dict(row) if row else None

def get_batch_count():
    """Get total number of batches."""
    with get_db() as conn:
        return conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]

def get_batches_dataframe():
    """Return batch data as a pandas DataFrame (for compatibility with existing engines)."""
    import pandas as pd
    with get_db() as conn:
        df = pd.read_sql_query("SELECT * FROM batches ORDER BY Batch_ID", conn)
    # Remove the created_at column added by DB
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    return df

# ═══════════════════════════════════════════════════════════
# PROPOSAL CRUD OPERATIONS
# ═══════════════════════════════════════════════════════════

def create_proposal(proposal_id, mode, batch_id, new_params, new_scores, improvement_pct):
    """Insert a new proposal into the database."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO proposals 
               (proposal_id, mode, batch_id, new_params, new_scores, 
                improvement_pct, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?)""",
            (
                proposal_id, mode, batch_id,
                json.dumps(new_params),
                json.dumps(new_scores),
                improvement_pct,
                datetime.now().isoformat(),
            )
        )

def get_pending_proposals_db():
    """Get all pending proposals."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM proposals WHERE status = 'PENDING'"
        ).fetchall()
        result = {}
        for row in rows:
            d = dict(row)
            d["new_params"] = json.loads(d["new_params"])
            d["new_scores"] = json.loads(d["new_scores"])
            result[d["proposal_id"]] = d
        return result

def get_proposal(proposal_id):
    """Get a single proposal."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["new_params"] = json.loads(d["new_params"])
            d["new_scores"] = json.loads(d["new_scores"])
            return d
        return None

def update_proposal_status(proposal_id, status, reviewer_note=""):
    """Update proposal status (ACCEPTED/REJECTED)."""
    with get_db() as conn:
        conn.execute(
            """UPDATE proposals 
               SET status = ?, reviewed_at = ?, reviewer_note = ?
               WHERE proposal_id = ?""",
            (status, datetime.now().isoformat(), reviewer_note, proposal_id)
        )

# ═══════════════════════════════════════════════════════════
# DECISIONS CRUD OPERATIONS
# ═══════════════════════════════════════════════════════════

def add_decision(proposal_id=None, action="", mode=None, batch_id=None,
                 improvement=None, reviewer_note=None, old_weights=None,
                 new_weights=None, reason=None):
    """Log a HITL decision."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO decisions 
               (proposal_id, action, mode, batch_id, improvement, 
                reviewer_note, old_weights, new_weights, reason, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                proposal_id, action, mode, batch_id, improvement,
                reviewer_note,
                json.dumps(old_weights) if old_weights else None,
                json.dumps(new_weights) if new_weights else None,
                reason,
                datetime.now().isoformat(),
            )
        )

def get_all_decisions():
    """Get all decisions."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY timestamp ASC"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d.get("old_weights"):
                d["old_weights"] = json.loads(d["old_weights"])
            if d.get("new_weights"):
                d["new_weights"] = json.loads(d["new_weights"])
            result.append(d)
        return result

def get_decisions_summary_db():
    """Get summary of all decisions."""
    decisions = get_all_decisions()
    accepted = sum(1 for d in decisions if d.get("action") == "ACCEPTED")
    rejected = sum(1 for d in decisions if d.get("action") == "REJECTED")
    reprioritized = sum(1 for d in decisions if d.get("action") == "REPRIORITIZED")
    return {
        "total": len(decisions),
        "accepted": accepted,
        "rejected": rejected,
        "reprioritized": reprioritized,
        "history": decisions,
    }

# ═══════════════════════════════════════════════════════════
# DATABASE INFO (for demo / judges)
# ═══════════════════════════════════════════════════════════

def get_db_stats():
    """Get database statistics for display."""
    with get_db() as conn:
        batch_count = conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        decision_count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        proposal_count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        pending_count = conn.execute("SELECT COUNT(*) FROM proposals WHERE status = 'PENDING'").fetchone()[0]
        
        # Get DB file size
        db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        
        return {
            "database": DB_PATH,
            "engine": "SQLite 3",
            "tables": ["batches", "decisions", "proposals"],
            "batch_count": batch_count,
            "decision_count": decision_count,
            "proposal_count": proposal_count,
            "pending_proposals": pending_count,
            "db_size_kb": round(db_size / 1024, 1),
        }


# ─── INIT ON IMPORT ──────────────────────────────────────
if __name__ == "__main__":
    init_db()
    stats = get_db_stats()
    print(f"\n📊 Database Stats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
