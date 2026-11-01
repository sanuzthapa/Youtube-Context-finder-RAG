"""FastAPI entrypoint and server runner for YouTube RAG.

Run directly via:
    python serve.py
Or via uvicorn:
    uvicorn serve:app --reload
"""

import sys
from pathlib import Path
import uvicorn

# Ensure the project root and ytrag package are in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.main import app, dynamic_query, DynamicQueryRequest, search, ask, dynamic_ingest_only, IngestOnlyRequest

if __name__ == "__main__":
    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=True)

