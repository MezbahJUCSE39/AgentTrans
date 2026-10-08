"""
Thin entry point for the LLM biomarker agent.

Validate without API:
    python agentic/examples/09_run_llm_biomarker_agent.py --validate-only

Run actual LLM:
    python agentic/examples/09_run_llm_biomarker_agent.py
"""
from pathlib import Path
import sys

AGENTIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTIC))

from llm.llm_biomarker_agent import main

if __name__ == "__main__":
    main()
