"""
Validate the biomarker MCP backend WITHOUT Claude Desktop and WITHOUT an LLM.

Run from project root:
    python agentic/examples/10_validate_mcp_biomarker_server.py
"""
from pathlib import Path
import json
import os
import sys

AGENTIC = Path(__file__).resolve().parents[1]
MCP_DIR = AGENTIC / "mcp"
sys.path.insert(0, str(MCP_DIR))

from biomarker_backend import BiomarkerBackend

PROJECT_ROOT = Path(
    os.environ.get("MEZBAH_PROJECT_ROOT", str(AGENTIC.parent))
).expanduser().resolve()

backend = BiomarkerBackend(PROJECT_ROOT)
print(json.dumps(backend.validate(), indent=2))
