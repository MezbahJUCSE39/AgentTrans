# Running the agent in Claude Desktop (MCP)

The agent is an MCP server that Claude Desktop launches as a local child process over stdio.
Nothing is sent to a third-party service except the conversation itself; all tool execution is
local.

## 1. Prerequisites

- Python 3.10+ (developed on 3.12)
- Claude Desktop
- the repository cloned, and the two PPMI-derived inputs rebuilt (see `data_access/README.md`)
  — strictly, the cached-replay path below runs without them, because the agent reads cached
  trainval results rather than recomputing

```bash
conda activate pdagent
pip install -U "mcp[cli]>=2,<3"
python -c "import mcp; print('MCP import OK')"
```

## 2. Validate the backend before involving any LLM

```bash
export MEZBAH_PROJECT_ROOT="/absolute/path/to/repo"
python agentic/examples/10_validate_mcp_biomarker_server.py
```

This exercises the backend directly: it loads the frozen config, cross-checks the thresholds
against the cached result files, walks the legal trajectory, and asserts that no code path exposes
held-out rows. A failure here is a configuration problem, not an LLM problem — fix it first.

## 3. Register the server

Edit Claude Desktop's config file:

- macOS `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "pd-biomarker-agent": {
      "command": "/absolute/path/to/conda/envs/pdagent/bin/python",
      "args": ["/absolute/path/to/repo/agentic/mcp/biomarker_mcp_server.py"],
      "env": { "MEZBAH_PROJECT_ROOT": "/absolute/path/to/repo" }
    }
  }
}
```

Use absolute paths for both the interpreter and the script — Claude Desktop does not inherit your
shell environment. Restart the app completely; the server appears in the tools menu.

## 4. Run

Paste `agentic/prompts/claude_biomarker_agent_prompt.md` as a single message. The expected
trajectory is seven calls:

```text
get_locked_objective
get_external_evidence_summary
get_initial_evidence_candidate
evaluate_panel(269)        -> FAIL_PRESERVATION_CONSTRAINT
run_feedback_rescue(269)
evaluate_panel(279)        -> PASS_BOTH
finalize_panel(279, "...") -> FINALIZED
```

Artifacts are written to `agentic/results/claude_mcp_agent/`.

## 5. Reproduce the guard-probe evaluation

Ask the model to attempt the nine violations listed in the main README before finishing legally.
Each must return `REFUSED` with a named reason. If any probe succeeds, the backend has regressed —
that is the regression test for this project.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Server does not appear | config JSON invalid, or relative paths | validate the JSON, use absolute paths, restart fully |
| `FileNotFoundError` on startup | `MEZBAH_PROJECT_ROOT` unset or wrong | set it in the `env` block, not only in your shell |
| Threshold mismatch exception | cached results and config disagree | re-run stage S4, or restore the frozen config |
| Tools present but every call refused | run not started | the first call of a run must be `get_locked_objective` |
| Output garbled | something printed to stdout | stdout is the MCP transport — log to stderr only |
