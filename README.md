# ⚡ AuraOptima — Golden Signature Intelligence Platform

> **AI-powered pharmaceutical batch manufacturing optimization using multi-objective Pareto analysis, human-in-the-loop governance, and Gemini-powered intelligent insights.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange?logo=google)](https://ai.google.dev)
[![SQLite](https://img.shields.io/badge/SQLite-Database-lightblue?logo=sqlite)](https://sqlite.org)

---

## 🎯 Problem Statement

Pharmaceutical manufacturing faces a **multi-objective optimization challenge**: how to simultaneously maximize **tablet quality**, maximize **production yield**, and minimize **energy consumption & carbon emissions** — all while maintaining regulatory compliance.

Traditional approaches optimize one metric at a time, leading to suboptimal tradeoffs. AuraOptima solves this using **Golden Signatures** — benchmark parameter profiles derived from Pareto-optimal batches.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│   Dashboard │ Batch Monitor │ Simulator │ ROI │ HITL     │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────────────┐
│                 FastAPI Backend                           │
│  ┌──────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │Deviation │ │   Golden     │ │  LLM Assistant      │  │
│  │ Engine   │ │  Signature   │ │  (Gemini + Ollama)  │  │
│  │          │ │   Engine     │ │                     │  │
│  └────┬─────┘ └──────┬───────┘ └──────────┬──────────┘  │
│       │              │                     │             │
│  ┌────▼──────────────▼─────────────────────▼──────────┐  │
│  │          HITL Manager (Human-in-the-Loop)          │  │
│  │     Propose → Review → Accept/Reject → Audit       │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       │                                  │
│  ┌────────────────────▼───────────────────────────────┐  │
│  │           SQLite Database + Data Pipeline           │  │
│  │     60 Batches │ Decisions │ Proposals │ Sigs       │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|------------|
| 🧬 **Golden Signatures** | Multi-objective Pareto-optimal benchmark profiles for 3 modes: Quality, Energy, Balanced |
| 📊 **Batch Monitor** | Real-time deviation analysis against golden signatures with severity color-coding |
| 🔮 **What-If Simulator** | Predict quality, yield, energy for hypothetical parameter combinations |
| 🧑‍🔬 **HITL Manager** | Human-in-the-loop governance: propose, accept/reject signature updates with audit trail |
| 🤖 **AI Chatbot** | Gemini-powered conversational assistant with domain knowledge and context awareness |
| 💰 **ROI Calculator** | Cost savings analysis if underperforming batches are optimized |
| 🌿 **Sustainability Dashboard** | Track energy & carbon footprint with SDG alignment |
| 📄 **PDF Reports** | Professional branded reports with charts, recommendations, and batch data |
| 🗄️ **SQLite Database** | Persistent storage with auto-seeding, WAL mode, and proper indexes |
| 🔔 **Alert System** | Real-time notifications for critical batches and pending proposals |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Google Gemini API Key](https://ai.google.dev/) (free tier available)

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd hackathonPrototype

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
# Edit .env file and add your Gemini API key:
# GEMINI_API_KEY=your-actual-api-key

# 5. Run the application
python app.py
```

Open **http://localhost:8000** in your browser.

---

## 📁 Project Structure

```
hackathonPrototype/
├── app.py                    # Application entry point
├── api.py                    # FastAPI routes (15+ endpoints)
├── database.py               # SQLite database layer with auto-seeding
├── data_pipeline.py          # Raw data → master DataFrame pipeline
├── golden_signature_engine.py # Pareto-optimal signature generation
├── deviation_engine.py       # Batch vs golden signature analysis
├── hitl_manager.py           # Human-in-the-loop proposal workflow
├── llm_assistant.py          # Gemini AI chatbot with domain context
├── report_generator.py       # Professional PDF report generation
├── index.html                # Single-file React frontend (2400+ lines)
├── test_all.py               # API endpoint test suite
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed)
└── data/
    ├── master_df.csv          # Processed batch data (60 batches)
    ├── golden_signatures.json # Generated golden signatures
    ├── auraoptima.db          # SQLite database
    └── *.xlsx                 # Raw production & process data
```

---

## 🔬 Technical Highlights

### Multi-Objective Optimization
- **Pareto Front Analysis**: Identifies non-dominated solutions across quality, yield, and energy
- **3 Optimization Modes**: Quality-first, Energy-first, Balanced — each with unique weight profiles
- **Composite Scoring**: Weighted normalized scores for fair cross-metric comparison

### AI / LLM Integration
- **Google Gemini API**: Primary LLM for chatbot and batch insights
- **Ollama Fallback**: Local model fallback for offline/rate-limited scenarios
- **Context-Aware Retrieval**: Dynamically injects relevant batch data based on query keywords
- **Rich System Prompt**: Full domain knowledge including fleet stats and golden signatures

### Responsible AI (HITL)
- **Proposal Workflow**: AI can suggest signature updates, but humans must approve
- **Audit Trail**: Every decision (accept/reject/reprioritize) is logged with timestamps
- **Weight Reprioritization**: Engineers can adjust optimization priorities with reason tracking

### Data Pipeline
- Processes **60 batches** from raw Excel time-series data
- Aggregates per-phase energy consumption (7 production phases)
- Engineering features: Quality Score, Yield Score, Energy Efficiency, Performance Score
- Carbon footprint calculation using India's emission factor (0.82 kg CO₂/kWh)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend |
| GET | `/batches` | All batch data |
| GET | `/golden-signatures` | Golden signature profiles |
| GET | `/deviation/{batch_id}` | Deviation analysis report |
| GET | `/rankings` | All batches ranked by deviation |
| POST | `/optimize` | Get optimal parameters for a mode |
| GET | `/fleet-health` | Fleet health summary |
| POST | `/simulate` | What-If parameter simulation |
| GET | `/roi` | ROI / cost savings analysis |
| GET | `/sustainability` | Energy & carbon overview |
| POST | `/chat` | AI chatbot conversation |
| GET | `/insights/{batch_id}` | AI-generated batch analysis |
| GET | `/report/{batch_id}` | Download PDF report |
| GET | `/proposals` | Pending HITL proposals |
| POST | `/approve-update` | Accept/reject a proposal |
| GET | `/decisions` | HITL decisions history |
| GET | `/db-stats` | Database statistics |

---

## 🌍 SDG Alignment

- **SDG 7** — Affordable & Clean Energy (energy optimization)
- **SDG 9** — Industry, Innovation & Infrastructure (smart manufacturing)
- **SDG 12** — Responsible Consumption & Production (waste reduction)
- **SDG 13** — Climate Action (carbon footprint reduction)

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Database**: SQLite with WAL mode
- **AI/ML**: Google Gemini API, scikit-learn, NumPy, Pandas
- **Frontend**: React 18 (CDN), Plotly.js, vanilla CSS
- **Reports**: fpdf2
- **Data**: OpenPyXL for Excel processing

---

## 👥 Team

Built for the hackathon by the AuraOptima team.

---

*© 2026 AuraOptima — Golden Signature Intelligence Platform*
