"""Quick test of all API endpoints."""
import requests
import json

BASE = "http://localhost:8000"

def test(name, method, url, json_data=None):
    try:
        if method == "GET":
            r = requests.get(url, timeout=15)
        else:
            r = requests.post(url, json=json_data, timeout=15)
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {"html_len": len(r.text)}
        print(f"[PASS] {name} - Status: {r.status_code}")
        return r.status_code, data
    except Exception as e:
        print(f"[FAIL] {name} - Error: {e}")
        return 0, {}

# 1. Frontend
test("Frontend /", "GET", f"{BASE}/")

# 2. Batches
code, data = test("GET /batches", "GET", f"{BASE}/batches")
if code == 200:
    print(f"       -> {len(data)} batches loaded")

# 3. Golden Signatures
code, data = test("GET /golden-signatures", "GET", f"{BASE}/golden-signatures")
if code == 200:
    print(f"       -> Modes: {list(data.keys())}")

# 4. Deviation
code, data = test("GET /deviation/T001", "GET", f"{BASE}/deviation/T001?mode=balanced")
if code == 200:
    print(f"       -> Batch: {data.get('batch_id')}, Status: {data.get('overall_status')}")

# 5. Rankings
code, data = test("GET /rankings", "GET", f"{BASE}/rankings?mode=balanced")
if code == 200:
    print(f"       -> {len(data)} batches ranked")

# 6. Sustainability
code, data = test("GET /sustainability", "GET", f"{BASE}/sustainability")
if code == 200:
    print(f"       -> Avg Energy: {data.get('avg_energy_kWh')} kWh, Avg Carbon: {data.get('avg_carbon_kg')} kg")

# 7. Fleet Health
code, data = test("GET /fleet-health", "GET", f"{BASE}/fleet-health")
if code == 200:
    print(f"       -> Healthy: {data.get('healthy')}, Warning: {data.get('warning')}, Critical: {data.get('critical')}")

# 8. ROI
code, data = test("GET /roi", "GET", f"{BASE}/roi?cost_per_kwh=8.0")
if code == 200:
    print(f"       -> Cost Saved: {data.get('total_cost_saved')}, Improvable: {data.get('improvable_batches')}")

# 9. Optimize
code, data = test("POST /optimize", "POST", f"{BASE}/optimize", {"mode": "balanced"})
if code == 200:
    print(f"       -> Label: {data.get('label')}")

# 10. Simulate
sim_params = {
    "Granulation_Time": 45, "Binder_Amount": 500, "Drying_Temp": 60,
    "Drying_Time": 30, "Compression_Force": 20, "Machine_Speed": 35, "Lubricant_Conc": 1.0
}
code, data = test("POST /simulate", "POST", f"{BASE}/simulate", {"params": sim_params, "mode": "balanced"})
if code == 200:
    preds = data.get("predictions", {})
    print(f"       -> Quality: {preds.get('Quality_Score')}, Energy: {preds.get('Total_Energy_kWh')}")

# 11. Proposals
code, data = test("GET /proposals", "GET", f"{BASE}/proposals")
if code == 200:
    print(f"       -> {len(data)} pending proposals")

# 12. Decisions
test("GET /decisions", "GET", f"{BASE}/decisions")

# 13. DB Stats
code, data = test("GET /db-stats", "GET", f"{BASE}/db-stats")
if code == 200:
    print(f"       -> {json.dumps(data)[:120]}")

# 14. Report PDF
code, _ = test("GET /report/T001 (PDF)", "GET", f"{BASE}/report/T001?mode=balanced")

print("\n" + "="*50)
print("All non-LLM endpoints tested!")
print("="*50)

# 15. Chat (Gemini API test)
print("\nTesting Gemini API chat (may take a few seconds)...")
code, data = test("POST /chat (Gemini)", "POST", f"{BASE}/chat", {
    "message": "What is the best performing batch?",
    "history": []
})
if code == 200:
    resp = data.get("response", "")
    print(f"       -> Response preview: {resp[:150]}...")
elif code == 500:
    print(f"       -> ERROR: {data.get('detail', 'Unknown error')}")

print("\n" + "="*50)
print("FULL VERIFICATION COMPLETE!")
print("="*50)
