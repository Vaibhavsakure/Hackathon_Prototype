"""
AuraOptima — Golden Signature Intelligence Platform
════════════════════════════════════════════════════
Entry point for the AuraOptima application.
Run:  python app.py
"""

import os
import uvicorn
from api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("=" * 55)
    print("   AuraOptima — Golden Signature Intelligence")
    print(f"   Starting server on http://0.0.0.0:{port}")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=port)
