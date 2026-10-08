"""
LLM Biomarker Decision Agent
============================

Purpose
-------
Add an actual LLM orchestration layer above the deterministic transcriptomics
pipeline that has already been executed.

The LLM NEVER:
- reads held-out test data,
- calculates ML metrics itself,
- changes feature-selection results,
- changes the locked BA/AUROC thresholds,
- invents PubMed/STRING evidence,
- directly edits the gene list.

Instead, the LLM chooses among a small set of deterministic tools that expose
the already-computed trainval evidence and rescue results.

Expected high-level trajectory
------------------------------
1. Inspect locked objective.
2. Inspect external-evidence summary.
3. Inspect the initial hard-evidence candidate (269 genes).
4. Evaluate 269 -> fails.
5. Request feedback rescue.
6. Evaluate 279 -> passes.
7. Finalize 279 -> STOP.

Important methodological note
-----------------------------
This script is an LLM-controller REPRODUCTION over the real cached outputs of
the completed deterministic pipeline. It does not pretend that the LLM created
the historical PubMed, STRING, or ML outputs retroactively.

OpenAI API
----------
Uses the Responses API and function calling.

Environment
-----------
OPENAI_API_KEY : required
OPENAI_MODEL   : optional; defaults to a dated model snapshot for reproducibility

Run
---
python agentic/examples/09_run_llm_biomarker_agent.py
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"
MAX_AGENT_TURNS = 12

REFERENCE_BA = 0.6512178588125292
REFERENCE_AUC = 0.7000514258999532
MIN_BA = 0.6312178588125292
MIN_AUC = 0.6800514258999532


SYSTEM_PROMPT = f"""
You are an LLM controller for a Parkinson's disease transcriptomic biomarker
refinement pipeline.

Your role is ORCHESTRATION, not calculation.

GOAL
Select the smallest evidence-guided gene panel that satisfies BOTH locked
trainval preservation constraints relative to the 593-gene reference.

LOCKED REFERENCE
- 593-gene balanced accuracy: {REFERENCE_BA:.12f}
- 593-gene AUROC: {REFERENCE_AUC:.12f}

LOCKED ACCEPTANCE THRESHOLDS
- balanced accuracy >= {MIN_BA:.12f}
- AUROC >= {MIN_AUC:.12f}

NON-NEGOTIABLE RULES
1. Never request, infer, or use held-out test data.
2. Use only values returned by tools.
3. Never invent a metric, gene, PubMed result, STRING result, or panel.
4. Never change the thresholds.
5. Do not directly add/remove genes. Panel construction is deterministic.
6. If the initial evidence-guided panel fails either threshold, request
   feedback-guided rescue.
7. After rescue, evaluate the smallest plausible rescued candidate before
   considering larger candidates.
8. If a candidate satisfies both thresholds, finalize the SMALLEST passing
   candidate allowed by the deterministic tool results.
9. If no reduced panel passes, retain the 593-gene reference.
10. Do not expose private chain-of-thought. In the final response give only a
    concise decision rationale based on observable tool outputs.
11. You MUST call finalize_panel before ending the run.

RECOMMENDED TOOL SEQUENCE
get_locked_objective
-> get_external_evidence_summary
-> get_initial_evidence_candidate
-> evaluate_panel(269)
-> run_feedback_rescue(269)
-> evaluate_panel(279)
-> finalize_panel(279)

You may deviate only if a tool result logically requires it.
"""


USER_TASK = """
Execute the biomarker refinement decision using the available deterministic
tools. Establish the objective, inspect the evidence-guided starting panel,
evaluate it, request rescue if required, and stop only after a valid final
panel has been finalized.

Return a concise final summary with:
- final action,
- panel size,
- observed BA and AUROC,
- thresholds,
- why the agent stopped,
- confirmation that held-out test data were not used.
"""


TOOLS = [
    {
        "type": "function",
        "name": "get_locked_objective",
        "description": (
            "Return the locked 593-gene trainval reference, preservation "
            "thresholds, and test-set prohibition. Call this before making "
            "any panel decision."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_external_evidence_summary",
        "description": (
            "Return a deterministic summary of the already-computed PubMed "
            "and STRING evidence across the 593-gene candidate universe. "
            "This does not access the internet."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_initial_evidence_candidate",
        "description": (
            "Return the initial hard evidence-guided candidate panel selected "
            "before feedback rescue. This candidate is deterministic and "
            "derived from the prior pipeline."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "evaluate_panel",
        "description": (
            "Return previously computed TRAIN/VALIDATION metrics for one "
            "allowed panel size. No held-out test data are accessible. "
            "Use this to decide whether a panel meets both locked thresholds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "panel_size": {
                    "type": "integer",
                    "enum": [269, 279, 289, 299, 309, 319, 593],
                    "description": "Allowed candidate panel size.",
                }
            },
            "required": ["panel_size"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "run_feedback_rescue",
        "description": (
            "Expose the deterministic feedback-rescue search results after "
            "the failed 269-gene hard-pruned panel. This tool can be called "
            "only after the 269-gene panel has been evaluated and shown to "
            "fail the locked constraints."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "current_panel_size": {
                    "type": "integer",
                    "enum": [269],
                    "description": "The failed hard-evidence seed panel.",
                }
            },
            "required": ["current_panel_size"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "finalize_panel",
        "description": (
            "Commit the final panel. The function enforces that the requested "
            "panel has been explicitly evaluated by the LLM run, passes both "
            "locked thresholds, and is the smallest passing rescued panel. "
            "If no reduced panel passes, 593 may be retained."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "panel_size": {
                    "type": "integer",
                    "enum": [279, 289, 299, 309, 319, 593],
                    "description": "Panel to commit as the agent's final decision.",
                }
            },
            "required": ["panel_size"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


@dataclass
class AgentState:
    objective_seen: bool = False
    evidence_seen: bool = False
    initial_candidate_seen: bool = False
    evaluated_sizes: set[int] = field(default_factory=set)
    rescue_seen: bool = False
    finalized: bool = False
    final_panel_size: int | None = None
    final_result: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_first(candidates: list[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    attempted = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not locate {label}. Tried:\n{attempted}"
    )


class ProjectFiles:
    """
    Resolve files from the user's existing project layout.

    The code intentionally exposes only cached TRAIN/VALIDATION outputs to the
    LLM controller. There is no path resolver for held-out predictions/results.
    """
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.agentic = project_root / "agentic"

        self.combined_evidence = find_first(
            [
                self.agentic / "results" / "string_consensus_comparison"
                / "combined_pubmed_string_593_gene_evidence.csv",
                self.agentic / "data" / "combined_pubmed_string_593_gene_evidence.csv",
                project_root / "combined_pubmed_string_593_gene_evidence.csv",
            ],
            "combined PubMed+STRING 593-gene evidence",
        )

        self.stage5_results = find_first(
            [
                self.agentic / "results" / "evidence_refinement"
                / "evidence_guided_panel_results.csv",
                self.agentic / "data" / "evidence_guided_panel_results.csv",
                project_root / "evidence_guided_panel_results.csv",
            ],
            "hard evidence-guided panel results",
        )

        self.stage5_selected = find_first(
            [
                self.agentic / "results" / "evidence_refinement"
                / "selected_evidence_panel.json",
                self.agentic / "data" / "selected_evidence_panel.json",
                project_root / "selected_evidence_panel.json",
            ],
            "selected hard evidence-guided panel JSON",
        )

        self.rescue_results = find_first(
            [
                self.agentic / "results" / "feedback_rescue"
                / "feedback_rescue_results.csv",
                self.agentic / "data" / "feedback_rescue_results.csv",
                project_root / "feedback_rescue_results.csv",
            ],
            "feedback rescue results",
        )

        self.rescue_selected = find_first(
            [
                self.agentic / "results" / "feedback_rescue"
                / "selected_feedback_rescue_panel.json",
                self.agentic / "data" / "selected_feedback_rescue_panel.json",
                project_root / "selected_feedback_rescue_panel.json",
            ],
            "selected feedback rescue panel JSON",
        )

        self.final_panel = find_first(
            [
                self.agentic / "results" / "feedback_rescue"
                / "FINAL_FEEDBACK_RESCUE_PANEL.csv",
                self.agentic / "data" / "FINAL_FEEDBACK_RESCUE_PANEL.csv",
                project_root / "FINAL_FEEDBACK_RESCUE_PANEL.csv",
            ],
            "final 279-gene feedback rescue panel",
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "combined_evidence": str(self.combined_evidence),
            "stage5_results": str(self.stage5_results),
            "stage5_selected": str(self.stage5_selected),
            "rescue_results": str(self.rescue_results),
            "rescue_selected": str(self.rescue_selected),
            "final_panel": str(self.final_panel),
        }


class BiomarkerToolbox:
    def __init__(self, files: ProjectFiles):
        self.files = files
        self.state = AgentState()

        self.evidence = pd.read_csv(files.combined_evidence)
        self.stage5 = pd.read_csv(files.stage5_results)
        self.rescue = pd.read_csv(files.rescue_results)
        self.final_panel = pd.read_csv(files.final_panel)

        with files.stage5_selected.open(encoding="utf-8") as f:
            self.stage5_selected = json.load(f)

        with files.rescue_selected.open(encoding="utf-8") as f:
            self.rescue_selected = json.load(f)

        self._validate_inputs()

    def _validate_inputs(self):
        # Structural checks.
        if len(self.evidence) != 593:
            raise ValueError(
                f"Expected 593 genes in combined evidence, found {len(self.evidence)}"
            )

        if len(self.final_panel) != 279:
            raise ValueError(
                f"Expected 279 genes in final panel, found {len(self.final_panel)}"
            )

        if self.stage5_selected.get("selected_panel_size") != 269:
            raise ValueError(
                "Stage-5 selected panel is not the expected 269-gene seed."
            )

        if self.rescue_selected.get("selected_panel_size") != 279:
            raise ValueError(
                "Feedback-rescue selected panel is not 279 genes."
            )

        if self.stage5_selected.get("heldout_test_used_for_selection") is not False:
            raise ValueError("Stage-5 JSON indicates held-out use.")

        if self.rescue_selected.get("heldout_test_used_for_selection") is not False:
            raise ValueError("Rescue JSON indicates held-out use.")

        # Verify every development result explicitly states no test access.
        for name, df in [
            ("stage5", self.stage5),
            ("rescue", self.rescue),
        ]:
            if "test_rows_accessed" not in df.columns:
                raise ValueError(f"{name} lacks test_rows_accessed.")
            if (pd.to_numeric(df["test_rows_accessed"], errors="coerce") != 0).any():
                raise ValueError(
                    f"{name} contains a row with test_rows_accessed != 0."
                )

        # Threshold consistency.
        thresholds = self.rescue_selected["thresholds"]
        if abs(float(thresholds["balanced_accuracy_min"]) - MIN_BA) > 1e-12:
            raise ValueError("BA threshold mismatch.")
        if abs(float(thresholds["roc_auc_min"]) - MIN_AUC) > 1e-12:
            raise ValueError("AUROC threshold mismatch.")

    def _result_for_size(self, panel_size: int) -> dict[str, Any]:
        if panel_size == 593:
            return {
                "panel_name": "locked_reference_593",
                "panel_size": 593,
                "balanced_accuracy_mean": REFERENCE_BA,
                "roc_auc_mean": REFERENCE_AUC,
                "passes_ba": True,
                "passes_auc": True,
                "passes_both": True,
                "data_scope": "trainval_only",
                "test_rows_accessed": 0,
                "role": "locked_reference",
            }

        # Prefer rescue table for 269-319 because it has a uniform schema.
        row = self.rescue[
            pd.to_numeric(self.rescue["panel_size"], errors="coerce") == panel_size
        ]
        if row.empty and panel_size == 269:
            row = self.stage5[
                pd.to_numeric(self.stage5["panel_size"], errors="coerce") == 269
            ]

        if row.empty:
            raise ValueError(f"No trainval result found for panel size {panel_size}.")

        r = row.iloc[0].to_dict()

        # Normalize booleans.
        pass_ba = float(r["balanced_accuracy_mean"]) >= MIN_BA
        pass_auc = float(r["roc_auc_mean"]) >= MIN_AUC

        return {
            "panel_name": str(r["panel_name"]),
            "panel_size": int(r["panel_size"]),
            "balanced_accuracy_mean": float(r["balanced_accuracy_mean"]),
            "balanced_accuracy_sd": float(r["balanced_accuracy_sd"]),
            "roc_auc_mean": float(r["roc_auc_mean"]),
            "roc_auc_sd": float(r["roc_auc_sd"]),
            "mcc_mean": float(r["mcc_mean"]),
            "sensitivity_pd_mean": float(r["sensitivity_pd_mean"]),
            "specificity_hc_mean": float(r["specificity_hc_mean"]),
            "f1_macro_mean": float(r["f1_macro_mean"]),
            "pr_auc_pd_mean": float(r["pr_auc_pd_mean"]),
            "passes_ba": bool(pass_ba),
            "passes_auc": bool(pass_auc),
            "passes_both": bool(pass_ba and pass_auc),
            "data_scope": str(r["data_scope"]),
            "test_rows_accessed": int(r["test_rows_accessed"]),
        }

    def get_locked_objective(self) -> dict[str, Any]:
        self.state.objective_seen = True
        return {
            "goal": (
                "Find the smallest evidence-guided reduced panel that preserves "
                "trainval predictive performance within the locked tolerances."
            ),
            "locked_reference": {
                "panel_size": 593,
                "balanced_accuracy_mean": REFERENCE_BA,
                "roc_auc_mean": REFERENCE_AUC,
            },
            "locked_thresholds": {
                "balanced_accuracy_min": MIN_BA,
                "roc_auc_min": MIN_AUC,
            },
            "heldout_policy": {
                "allowed_during_agent_selection": False,
                "test_tool_exposed_to_agent": False,
            },
        }

    def get_external_evidence_summary(self) -> dict[str, Any]:
        if not self.state.objective_seen:
            return {
                "status": "REFUSED",
                "reason": "Call get_locked_objective first.",
            }

        self.state.evidence_seen = True

        pub = int(
            pd.to_numeric(
                self.evidence["pubmed_evidence_flag"], errors="coerce"
            ).fillna(0).sum()
        )
        string = int(
            pd.to_numeric(
                self.evidence["string_network_flag"], errors="coerce"
            ).fillna(0).sum()
        )
        dual = int(
            (
                (
                    pd.to_numeric(
                        self.evidence["pubmed_evidence_flag"], errors="coerce"
                    ).fillna(0)
                    == 1
                )
                & (
                    pd.to_numeric(
                        self.evidence["string_network_flag"], errors="coerce"
                    ).fillna(0)
                    == 1
                )
            ).sum()
        )

        group_counts = (
            self.evidence["fast_track_evidence_group"]
            .fillna("UNKNOWN")
            .value_counts()
            .to_dict()
        )

        return {
            "candidate_universe": 593,
            "pubmed_positive_genes": pub,
            "string_connected_genes": string,
            "pubmed_and_string_genes": dual,
            "evidence_group_counts": {
                str(k): int(v) for k, v in group_counts.items()
            },
            "interpretation_boundary": (
                "PubMed hits are coarse retrieval signals and STRING edges are "
                "functional associations; neither is a causal or clinical score."
            ),
        }

    def get_initial_evidence_candidate(self) -> dict[str, Any]:
        if not self.state.evidence_seen:
            return {
                "status": "REFUSED",
                "reason": "Inspect external evidence before requesting a candidate.",
            }

        self.state.initial_candidate_seen = True

        return {
            "status": "OK",
            "panel_name": self.stage5_selected["selected_panel"],
            "panel_size": int(self.stage5_selected["selected_panel_size"]),
            "construction": (
                "All 41 genes with 5/5 selector support plus all genes jointly "
                "supported by PubMed and the STRING 593-gene network."
            ),
            "stage5_selection_note": self.stage5_selected["selection_reason"],
            "requires_predictive_evaluation": True,
        }

    def evaluate_panel(self, panel_size: int) -> dict[str, Any]:
        if not self.state.initial_candidate_seen:
            return {
                "status": "REFUSED",
                "reason": "Call get_initial_evidence_candidate first.",
            }

        result = self._result_for_size(panel_size)

        if result["test_rows_accessed"] != 0:
            raise RuntimeError(
                "Safety violation: a purported trainval result accessed test rows."
            )

        self.state.evaluated_sizes.add(panel_size)

        return {
            "status": "OK",
            **result,
            "locked_thresholds": {
                "balanced_accuracy_min": MIN_BA,
                "roc_auc_min": MIN_AUC,
            },
            "decision_signal": (
                "PASS_BOTH"
                if result["passes_both"]
                else "FAIL_PRESERVATION_CONSTRAINT"
            ),
        }

    def run_feedback_rescue(self, current_panel_size: int) -> dict[str, Any]:
        if current_panel_size != 269:
            return {
                "status": "REFUSED",
                "reason": "Feedback rescue is defined from the 269-gene seed.",
            }

        if 269 not in self.state.evaluated_sizes:
            return {
                "status": "REFUSED",
                "reason": "Evaluate the 269-gene panel before requesting rescue.",
            }

        seed_result = self._result_for_size(269)
        if seed_result["passes_both"]:
            return {
                "status": "REFUSED",
                "reason": "269 already passes; rescue is not justified.",
            }

        self.state.rescue_seen = True

        rescue_rows = []
        for size in sorted(
            int(x)
            for x in pd.to_numeric(
                self.rescue["panel_size"], errors="coerce"
            ).dropna().unique()
            if int(x) > 269
        ):
            r = self._result_for_size(size)
            rescue_rows.append(
                {
                    "panel_size": size,
                    "balanced_accuracy_mean": r["balanced_accuracy_mean"],
                    "roc_auc_mean": r["roc_auc_mean"],
                    "mcc_mean": r["mcc_mean"],
                    "passes_ba": r["passes_ba"],
                    "passes_auc": r["passes_auc"],
                    "passes_both": r["passes_both"],
                }
            )

        return {
            "status": "OK",
            "rescue_policy": {
                "seed_size": 269,
                "priority": (
                    "Restore excluded genes with exactly one external source "
                    "before genes with neither source; within group use the "
                    "pre-existing reproducibility rank."
                ),
                "original_search_note": (
                    "The deterministic rescue script used a coarse-then-fine "
                    "search. The table below contains the actually evaluated "
                    "trainval candidates."
                ),
            },
            "evaluated_rescue_candidates": rescue_rows,
            "selection_rule": (
                "Choose the smallest candidate satisfying BOTH locked thresholds."
            ),
            "next_agent_action": (
                "Explicitly evaluate the smallest passing candidate before finalizing."
            ),
        }

    def finalize_panel(self, panel_size: int) -> dict[str, Any]:
        if panel_size != 593 and not self.state.rescue_seen:
            return {
                "status": "REFUSED",
                "reason": "Reduced-panel finalization requires feedback rescue.",
            }

        if panel_size not in self.state.evaluated_sizes:
            return {
                "status": "REFUSED",
                "reason": (
                    f"Panel {panel_size} has not been explicitly evaluated by "
                    "this LLM run. Call evaluate_panel first."
                ),
            }

        requested = self._result_for_size(panel_size)

        # Determine the smallest passing rescued panel from the actual deterministic
        # rescue results. This is a guardrail: the LLM cannot choose a larger panel
        # simply because it prefers it.
        passing = []
        for size in sorted(
            int(x)
            for x in pd.to_numeric(
                self.rescue["panel_size"], errors="coerce"
            ).dropna().unique()
            if int(x) > 269
        ):
            r = self._result_for_size(size)
            if r["passes_both"]:
                passing.append(size)

        if passing:
            smallest_passing = min(passing)
            if panel_size != smallest_passing:
                return {
                    "status": "REFUSED",
                    "reason": (
                        f"Panel {panel_size} is not the smallest passing rescue "
                        f"candidate. Smallest passing panel is {smallest_passing}."
                    ),
                }
        else:
            # If no reduced panel passes, only 593 may be finalized.
            if panel_size != 593:
                return {
                    "status": "REFUSED",
                    "reason": (
                        "No reduced candidate passes both constraints; retain 593."
                    ),
                }

        if panel_size != 593 and not requested["passes_both"]:
            return {
                "status": "REFUSED",
                "reason": "Requested reduced panel does not pass both thresholds.",
            }

        if requested["test_rows_accessed"] != 0:
            raise RuntimeError("Safety violation during finalization.")

        self.state.finalized = True
        self.state.final_panel_size = panel_size
        self.state.final_result = requested

        return {
            "status": "FINALIZED",
            "action": (
                "ACCEPT_REDUCED_PANEL"
                if panel_size < 593
                else "RETAIN_593_REFERENCE"
            ),
            "panel_size": panel_size,
            "balanced_accuracy_mean": requested["balanced_accuracy_mean"],
            "roc_auc_mean": requested["roc_auc_mean"],
            "thresholds": {
                "balanced_accuracy_min": MIN_BA,
                "roc_auc_min": MIN_AUC,
            },
            "stop_reason": (
                "This is the smallest deterministic rescue candidate that "
                "satisfies both locked trainval preservation constraints."
                if panel_size < 593
                else "No reduced panel satisfied both locked constraints."
            ),
            "heldout_test_used": False,
            "test_rows_accessed": 0,
        }

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "get_locked_objective":
            return self.get_locked_objective()
        if name == "get_external_evidence_summary":
            return self.get_external_evidence_summary()
        if name == "get_initial_evidence_candidate":
            return self.get_initial_evidence_candidate()
        if name == "evaluate_panel":
            return self.evaluate_panel(int(args["panel_size"]))
        if name == "run_feedback_rescue":
            return self.run_feedback_rescue(int(args["current_panel_size"]))
        if name == "finalize_panel":
            return self.finalize_panel(int(args["panel_size"]))
        raise KeyError(f"Unknown tool: {name}")


class AuditLog:
    def __init__(
        self,
        outdir: Path,
        model: str,
        files: ProjectFiles,
    ):
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = outdir / f"llm_agent_audit_{stamp}.jsonl"
        self.md_path = outdir / f"llm_agent_audit_{stamp}.md"
        self.summary_path = outdir / f"llm_agent_summary_{stamp}.json"
        self.events: list[dict[str, Any]] = []

        self.log(
            "run_start",
            {
                "timestamp_utc": utc_now(),
                "model": model,
                "execution_mode": "cached_trainval_evidence_reproduction",
                "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
                "user_task_sha256": sha256_text(USER_TASK),
                "input_files": {
                    key: {
                        "path": value,
                        "sha256": sha256_file(Path(value)),
                    }
                    for key, value in files.as_dict().items()
                },
                "heldout_test_tool_exposed": False,
            },
        )

    def log(self, event_type: str, payload: dict[str, Any]):
        event = {
            "event_index": len(self.events) + 1,
            "event_type": event_type,
            "timestamp_utc": utc_now(),
            "payload": payload,
        }
        self.events.append(event)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def finalize(
        self,
        state: AgentState,
        final_text: str,
        model: str,
    ):
        summary = {
            "model": model,
            "completed": bool(state.finalized),
            "final_panel_size": state.final_panel_size,
            "final_result": state.final_result,
            "evaluated_sizes_called_by_llm": sorted(state.evaluated_sizes),
            "heldout_test_tool_exposed": False,
            "heldout_test_used": False,
            "final_model_text": final_text,
            "audit_jsonl": str(self.jsonl_path),
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        lines = [
            "# LLM Biomarker Agent Audit",
            "",
            f"- **Model:** `{model}`",
            f"- **Completed:** `{state.finalized}`",
            f"- **Final panel:** `{state.final_panel_size}`",
            "- **Held-out test tool exposed:** `False`",
            "- **Held-out test used:** `False`",
            "",
            "## Tool trajectory",
            "",
        ]

        for e in self.events:
            if e["event_type"] == "tool_call":
                p = e["payload"]
                lines.append(
                    f"- Agent called `{p['name']}` with `{json.dumps(p['arguments'])}`"
                )
            elif e["event_type"] == "tool_result":
                p = e["payload"]
                result = p["result"]
                status = result.get("status", "")
                lines.append(
                    f"  - Result from `{p['name']}`: `{status}`"
                )

        lines += [
            "",
            "## Final agent response",
            "",
            final_text or "(No final text returned.)",
            "",
            "## Reproducibility note",
            "",
            (
                "This audit is an LLM-controller reproduction over cached, "
                "previously generated trainval evidence/results. The controller "
                "did not access held-out data and did not recalculate PubMed, "
                "STRING, or ML metrics."
            ),
        ]

        self.md_path.write_text("\n".join(lines), encoding="utf-8")


def run_agent(project_root: Path, model: str) -> tuple[AgentState, AuditLog, str]:
    files = ProjectFiles(project_root)
    toolbox = BiomarkerToolbox(files)

    audit_dir = project_root / "agentic" / "results" / "llm_agent"
    audit = AuditLog(audit_dir, model, files)

    client = OpenAI()

    input_items: list[Any] = [
        {
            "role": "user",
            "content": USER_TASK,
        }
    ]

    final_text = ""

    for turn in range(1, MAX_AGENT_TURNS + 1):
        audit.log("model_turn_start", {"turn": turn})

        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            tools=TOOLS,
            input=input_items,
        )

        # Preserve model output as required by the Responses tool-calling loop.
        input_items += response.output

        function_calls = [
            item for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]

        if function_calls:
            for item in function_calls:
                args = json.loads(item.arguments or "{}")
                audit.log(
                    "tool_call",
                    {
                        "turn": turn,
                        "name": item.name,
                        "arguments": args,
                        "call_id": item.call_id,
                    },
                )

                try:
                    result = toolbox.dispatch(item.name, args)
                except Exception as exc:
                    result = {
                        "status": "ERROR",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }

                audit.log(
                    "tool_result",
                    {
                        "turn": turn,
                        "name": item.name,
                        "call_id": item.call_id,
                        "result": result,
                    },
                )

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(
                            result,
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )

            continue

        # No function calls in this model turn.
        final_text = response.output_text or ""

        if toolbox.state.finalized:
            audit.log(
                "model_final_response",
                {
                    "turn": turn,
                    "text": final_text,
                },
            )
            break

        # Guard against early stopping before finalize_panel.
        audit.log(
            "early_stop_prevented",
            {
                "turn": turn,
                "text": final_text,
                "instruction": "Continue and call finalize_panel.",
            },
        )
        input_items.append(
            {
                "role": "user",
                "content": (
                    "You have not finalized a panel. Continue using the tools. "
                    "You MUST call finalize_panel before ending."
                ),
            }
        )
    else:
        raise RuntimeError(
            f"Agent exceeded MAX_AGENT_TURNS={MAX_AGENT_TURNS} without finalization."
        )

    if not toolbox.state.finalized:
        raise RuntimeError("Agent ended without a finalized panel.")

    audit.finalize(toolbox.state, final_text, model)
    return toolbox.state, audit, final_text


def validate_only(project_root: Path) -> dict[str, Any]:
    """
    Verify the complete deterministic state without calling an LLM/API.
    Useful before spending any API credits.
    """
    files = ProjectFiles(project_root)
    toolbox = BiomarkerToolbox(files)

    return {
        "status": "PASS",
        "project_root": str(project_root),
        "files": files.as_dict(),
        "combined_evidence_rows": len(toolbox.evidence),
        "final_panel_rows": len(toolbox.final_panel),
        "stage5_selected_size": toolbox.stage5_selected["selected_panel_size"],
        "rescue_selected_size": toolbox.rescue_selected["selected_panel_size"],
        "heldout_test_tool_exposed": False,
        "locked_thresholds": {
            "balanced_accuracy_min": MIN_BA,
            "roc_auc_min": MIN_AUC,
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing the agentic/ folder.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help=(
            "OpenAI model name. Default: OPENAI_MODEL env var, otherwise "
            f"{DEFAULT_MODEL}"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check files/guardrails without making an OpenAI API request.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()

    if args.validate_only:
        result = validate_only(project_root)
        print(json.dumps(result, indent=2))
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it in your shell; do not put "
            "the API key inside this Python file."
        )

    state, audit, final_text = run_agent(
        project_root=project_root,
        model=args.model,
    )

    print("\n" + "=" * 80)
    print("LLM BIOMARKER AGENT COMPLETE")
    print("=" * 80)
    print(f"Final panel size: {state.final_panel_size}")
    print(f"Model: {args.model}")
    print(f"Audit JSONL: {audit.jsonl_path}")
    print(f"Audit Markdown: {audit.md_path}")
    print(f"Summary JSON: {audit.summary_path}")
    print("\nFinal LLM response:")
    print(final_text)


if __name__ == "__main__":
    main()
