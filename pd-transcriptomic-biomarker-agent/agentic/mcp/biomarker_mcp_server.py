"""
Claude Desktop MCP server for the Mezbah Transcriptomics biomarker agent.

Current MCP Python SDK v2:
    pip install "mcp[cli]"

Claude Desktop launches this file over stdio.
DO NOT print to stdout; stdout is the MCP protocol transport.
"""
from __future__ import annotations

import logging

from mcp.server import MCPServer

from biomarker_backend import BiomarkerBackend, project_root_from_env


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mezbah-biomarker-mcp")

backend = BiomarkerBackend(project_root_from_env())

mcp = MCPServer(
    "Mezbah Transcriptomics Biomarker Agent Tools",
    version="1.0.0",
)


@mcp.tool()
def get_locked_objective() -> dict:
    """
    Start (or, after a completed run, restart) an audited biomarker-agent run.

    Returns the immutable 593-gene trainval reference, the locked balanced-
    accuracy/AUROC acceptance thresholds, and confirmation that no held-out
    test tool is available. Claude should call this first.
    """
    return backend.get_locked_objective()


@mcp.tool()
def get_external_evidence_summary() -> dict:
    """
    Inspect cached PubMed and STRING evidence for the 593-gene candidate set.

    Uses only previously generated development-stage evidence. It performs no
    web request and exposes no held-out test data.
    """
    return backend.get_external_evidence_summary()


@mcp.tool()
def get_initial_evidence_candidate() -> dict:
    """
    Return the deterministic initial hard evidence-guided candidate panel.

    The cached pipeline selected the 269-gene core-preserving evidence panel.
    Claude must evaluate it before requesting feedback rescue.
    """
    return backend.get_initial_evidence_candidate()


@mcp.tool()
def evaluate_panel(panel_size: int) -> dict:
    """
    Return cached TRAIN/VALIDATION metrics for an allowed panel.

    Allowed sizes are 269, 279, 289, 299, 309, 319, and the locked 593
    reference. Rescued panels cannot be evaluated until feedback rescue is
    requested. No held-out results are exposed.
    """
    return backend.evaluate_panel(panel_size)


@mcp.tool()
def run_feedback_rescue(current_panel_size: int) -> dict:
    """
    Invoke the deterministic feedback-rescue stage after the 269-gene seed fails.

    The tool exposes the already-computed trainval rescue trajectory and tells
    Claude which smallest passing candidate must be explicitly evaluated next.
    """
    return backend.run_feedback_rescue(current_panel_size)


@mcp.tool()
def finalize_panel(panel_size: int, decision_note: str) -> dict:
    """
    Commit the agent's final panel and finish the audited run.

    Guardrails require that the panel was explicitly evaluated during this
    Claude run and that it is the smallest cached rescued panel satisfying BOTH
    locked trainval thresholds. decision_note should be a concise observable-
    evidence rationale, not hidden chain-of-thought.

    On success, writes JSONL, Markdown, and summary JSON audit artifacts.
    """
    return backend.finalize_panel(panel_size, decision_note)


if __name__ == "__main__":
    # stdio is the default and is the correct transport for a local
    # Claude Desktop child process.
    mcp.run()
