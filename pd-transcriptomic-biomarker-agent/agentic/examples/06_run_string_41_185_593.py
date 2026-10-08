"""
Run comparative STRING analysis for the fixed 41 / 185 / 593 panels.

From project root:
    python agentic/examples/10_run_string_41_185_593.py
"""
from pathlib import Path
import sys

AGENTIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTIC))

from tools.string_consensus_comparison import main

if __name__ == "__main__":
    main()
