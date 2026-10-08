# Headless controller (OpenAI API)

`agentic/llm/llm_biomarker_agent.py` is the same decision loop without a desktop app: an
OpenAI-compatible tool-calling controller over the identical deterministic tool set. Use it for
batch reproduction, for ablations across models, or on a machine with no GUI.

## Run

```bash
pip install -U openai pandas
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-5.4-mini-2026-03-17"     # pin a dated snapshot for reproducibility
python agentic/examples/09_run_llm_biomarker_agent.py
```

Configuration lives in `agentic/config/llm_biomarker_agent_config.json`:

```json
{
  "execution_mode": "cached_trainval_evidence_reproduction",
  "locked_thresholds": { "balanced_accuracy_min": 0.6312178588, "roc_auc_min": 0.6800514259 },
  "allowed_panel_sizes": [269, 279, 289, 299, 309, 319, 593],
  "expected_final_panel": 279,
  "heldout_tool_exposed": false
}
```

`expected_final_panel` is recorded for post-run verification only; it is never shown to the model.

## What the controller may and may not do

| May | May not |
|---|---|
| read cached trainval evidence through functions | read held-out test results (no such function) |
| request a panel size from the frozen grid | change thresholds or the rescue order |
| call the rescue tool after the seed fails | add or remove individual genes |
| finalize the smallest passing panel | invent a metric or a citation |
| write a short observable-evidence rationale | end a run without `finalize_panel` |

## Scientific framing

State this explicitly in any write-up: the run is an **LLM-controller reproduction over cached,
previously computed trainval outputs**. The contribution is a governed decision procedure with a
complete audit trail — not autonomous discovery. Conflating the two is the single easiest way to
lose a reviewer's trust.

## Swapping models

Any OpenAI-compatible endpoint works. For a model comparison, keep the prompt, the config and the
cached state identical and vary only `OPENAI_MODEL`; then diff the resulting audit JSONL files.
Trajectory identity across models and hosts is the reproducibility measurement — in the runs
reported here, two different hosts produced the same seven-call path.
