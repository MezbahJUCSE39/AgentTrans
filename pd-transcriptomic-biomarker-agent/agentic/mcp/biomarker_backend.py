"""
Deterministic backend for the Mezbah Transcriptomics MCP biomarker agent.

This module contains NO LLM/API code and NO MCP transport code.
It exposes only cached TRAIN/VALIDATION evidence/results.

There is intentionally no tool or path resolver for held-out test results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import uuid

import pandas as pd


REFERENCE_BA = 0.6512178588125292
REFERENCE_AUC = 0.7000514258999532
MIN_BA = 0.6312178588125292
MIN_AUC = 0.6800514258999532


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Round-trip through JSON to normalize NumPy/pandas scalar types."""
    return json.loads(json.dumps(value, default=str))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_first(candidates: list[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p.resolve()
    attempted = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not locate {label}. Tried:\n{attempted}"
    )


class ProjectFiles:
    """Resolve only the cached development-stage files required by the agent."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.agentic = self.project_root / "agentic"

        self.combined_evidence = find_first(
            [
                self.agentic / "results" / "string_consensus_comparison"
                / "combined_pubmed_string_593_gene_evidence.csv",
                self.agentic / "data" / "combined_pubmed_string_593_gene_evidence.csv",
                self.project_root / "combined_pubmed_string_593_gene_evidence.csv",
            ],
            "combined PubMed+STRING 593-gene evidence",
        )

        self.stage5_results = find_first(
            [
                self.agentic / "results" / "evidence_refinement"
                / "evidence_guided_panel_results.csv",
                self.agentic / "data" / "evidence_guided_panel_results.csv",
                self.project_root / "evidence_guided_panel_results.csv",
            ],
            "hard evidence-guided panel results",
        )

        self.stage5_selected = find_first(
            [
                self.agentic / "results" / "evidence_refinement"
                / "selected_evidence_panel.json",
                self.agentic / "data" / "selected_evidence_panel.json",
                self.project_root / "selected_evidence_panel.json",
            ],
            "selected hard evidence-guided panel JSON",
        )

        self.rescue_results = find_first(
            [
                self.agentic / "results" / "feedback_rescue"
                / "feedback_rescue_results.csv",
                self.agentic / "data" / "feedback_rescue_results.csv",
                self.project_root / "feedback_rescue_results.csv",
            ],
            "feedback rescue results",
        )

        self.rescue_selected = find_first(
            [
                self.agentic / "results" / "feedback_rescue"
                / "selected_feedback_rescue_panel.json",
                self.agentic / "data" / "selected_feedback_rescue_panel.json",
                self.project_root / "selected_feedback_rescue_panel.json",
            ],
            "selected feedback rescue panel JSON",
        )

        self.final_panel = find_first(
            [
                self.agentic / "results" / "feedback_rescue"
                / "FINAL_FEEDBACK_RESCUE_PANEL.csv",
                self.agentic / "data" / "FINAL_FEEDBACK_RESCUE_PANEL.csv",
                self.project_root / "FINAL_FEEDBACK_RESCUE_PANEL.csv",
            ],
            "final feedback rescue panel",
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


@dataclass
class AgentState:
    run_id: str | None = None
    started_at_utc: str | None = None
    objective_seen: bool = False
    evidence_seen: bool = False
    initial_candidate_seen: bool = False
    evaluated_sizes: set[int] = field(default_factory=set)
    rescue_seen: bool = False
    finalized: bool = False
    final_panel_size: int | None = None
    final_result: dict[str, Any] | None = None
    final_decision_note: str | None = None


class AuditManager:
    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.run_id: str | None = None
        self.jsonl_path: Path | None = None
        self.md_path: Path | None = None
        self.summary_path: Path | None = None
        self.events: list[dict[str, Any]] = []

    def start(self, files: ProjectFiles) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        self.run_id = f"{stamp}_{suffix}"
        self.jsonl_path = self.outdir / f"claude_mcp_agent_audit_{self.run_id}.jsonl"
        self.md_path = self.outdir / f"claude_mcp_agent_audit_{self.run_id}.md"
        self.summary_path = self.outdir / f"claude_mcp_agent_summary_{self.run_id}.json"
        self.events = []

        self.log(
            "run_start",
            {
                "run_id": self.run_id,
                "execution_mode": "claude_desktop_mcp_cached_trainval_reproduction",
                "input_files": {
                    k: {"path": v, "sha256": sha256_file(Path(v))}
                    for k, v in files.as_dict().items()
                },
                "heldout_test_tool_exposed": False,
                "locked_reference": {
                    "panel_size": 593,
                    "balanced_accuracy_mean": REFERENCE_BA,
                    "roc_auc_mean": REFERENCE_AUC,
                },
                "locked_thresholds": {
                    "balanced_accuracy_min": MIN_BA,
                    "roc_auc_min": MIN_AUC,
                },
            },
        )
        return self.run_id

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.run_id is None or self.jsonl_path is None:
            return
        event = {
            "event_index": len(self.events) + 1,
            "event_type": event_type,
            "timestamp_utc": utc_now(),
            "payload": json_safe(payload),
        }
        self.events.append(event)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def finalize(self, state: AgentState) -> None:
        if (
            self.run_id is None
            or self.summary_path is None
            or self.md_path is None
        ):
            return

        summary = {
            "run_id": self.run_id,
            "completed": state.finalized,
            "final_panel_size": state.final_panel_size,
            "final_result": state.final_result,
            "final_decision_note": state.final_decision_note,
            "evaluated_sizes_called_by_claude": sorted(state.evaluated_sizes),
            "heldout_test_tool_exposed": False,
            "heldout_test_used": False,
            "audit_jsonl": str(self.jsonl_path),
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        lines = [
            "# Claude Desktop MCP Biomarker Agent Audit",
            "",
            f"- **Run ID:** `{self.run_id}`",
            f"- **Completed:** `{state.finalized}`",
            f"- **Final panel:** `{state.final_panel_size}`",
            f"- **Explicitly evaluated by Claude:** `{sorted(state.evaluated_sizes)}`",
            "- **Held-out test tool exposed:** `False`",
            "- **Held-out test used:** `False`",
            "",
            "## Tool trajectory",
            "",
        ]

        for event in self.events:
            if event["event_type"] == "tool_call":
                p = event["payload"]
                lines.append(
                    f"- Claude called `{p['tool']}` with "
                    f"`{json.dumps(p['arguments'], ensure_ascii=False)}`"
                )
            elif event["event_type"] == "tool_result":
                p = event["payload"]
                status = p["result"].get("status", "")
                signal = p["result"].get("decision_signal", "")
                extra = f" — {signal}" if signal else ""
                lines.append(
                    f"  - `{p['tool']}` returned `{status}`{extra}"
                )

        lines += [
            "",
            "## Finalization",
            "",
            f"- **Final panel:** {state.final_panel_size}",
            f"- **Decision note supplied by Claude:** {state.final_decision_note or '(none)'}",
            "",
            "## Reproducibility boundary",
            "",
            (
                "This is an LLM-orchestration reproduction over cached "
                "trainval-only evidence and model results. Claude Desktop "
                "did not recompute PubMed, STRING, or ML results and was not "
                "given a held-out test tool."
            ),
        ]
        self.md_path.write_text("\n".join(lines), encoding="utf-8")


class BiomarkerBackend:
    """
    Stateful deterministic backend used by the MCP tool wrappers.

    A new run starts when get_locked_objective is called after no prior run
    or after the previous run has finalized.
    """

    def __init__(self, project_root: Path):
        self.files = ProjectFiles(project_root)
        self.evidence = pd.read_csv(self.files.combined_evidence)
        self.stage5 = pd.read_csv(self.files.stage5_results)
        self.rescue = pd.read_csv(self.files.rescue_results)
        self.final_panel = pd.read_csv(self.files.final_panel)

        with self.files.stage5_selected.open(encoding="utf-8") as f:
            self.stage5_selected = json.load(f)
        with self.files.rescue_selected.open(encoding="utf-8") as f:
            self.rescue_selected = json.load(f)

        self._validate_inputs()

        self.state = AgentState()
        self.audit = AuditManager(
            self.files.agentic / "results" / "claude_mcp_agent"
        )

    def _validate_inputs(self) -> None:
        if len(self.evidence) != 593:
            raise ValueError(
                f"Expected 593 evidence rows; found {len(self.evidence)}."
            )
        if len(self.final_panel) != 279:
            raise ValueError(
                f"Expected 279 genes in final rescue panel; found {len(self.final_panel)}."
            )
        if int(self.stage5_selected.get("selected_panel_size", -1)) != 269:
            raise ValueError("Stage-5 selected panel must be 269 genes.")
        if int(self.rescue_selected.get("selected_panel_size", -1)) != 279:
            raise ValueError("Rescue selected panel must be 279 genes.")
        if self.stage5_selected.get("heldout_test_used_for_selection") is not False:
            raise ValueError("Stage-5 JSON indicates held-out test use.")
        if self.rescue_selected.get("heldout_test_used_for_selection") is not False:
            raise ValueError("Rescue JSON indicates held-out test use.")

        for name, df in [("stage5", self.stage5), ("rescue", self.rescue)]:
            if "test_rows_accessed" not in df.columns:
                raise ValueError(f"{name} lacks test_rows_accessed.")
            test_access = pd.to_numeric(
                df["test_rows_accessed"], errors="coerce"
            )
            if test_access.isna().any() or (test_access != 0).any():
                raise ValueError(
                    f"{name} contains nonzero/invalid test_rows_accessed."
                )

        t = self.rescue_selected["thresholds"]
        if abs(float(t["balanced_accuracy_min"]) - MIN_BA) > 1e-12:
            raise ValueError("Balanced-accuracy threshold mismatch.")
        if abs(float(t["roc_auc_min"]) - MIN_AUC) > 1e-12:
            raise ValueError("AUROC threshold mismatch.")

    def validate(self) -> dict[str, Any]:
        return {
            "status": "PASS",
            "project_root": str(self.files.project_root),
            "combined_evidence_rows": len(self.evidence),
            "final_panel_rows": len(self.final_panel),
            "stage5_selected_size": int(self.stage5_selected["selected_panel_size"]),
            "rescue_selected_size": int(self.rescue_selected["selected_panel_size"]),
            "heldout_test_tool_exposed": False,
            "locked_thresholds": {
                "balanced_accuracy_min": MIN_BA,
                "roc_auc_min": MIN_AUC,
            },
            "files": self.files.as_dict(),
        }

    def _active_run_required(self) -> dict[str, Any] | None:
        if self.state.run_id is None:
            return {
                "status": "REFUSED",
                "reason": "Call get_locked_objective first to start an audited run.",
            }
        if self.state.finalized:
            return {
                "status": "REFUSED",
                "reason": (
                    "This run is already finalized. Call get_locked_objective "
                    "to start a new audited run."
                ),
                "run_id": self.state.run_id,
            }
        return None

    def _log_call(
        self,
        tool: str,
        arguments: dict[str, Any],
        fn,
    ) -> dict[str, Any]:
        self.audit.log(
            "tool_call",
            {"tool": tool, "arguments": arguments},
        )
        try:
            result = fn()
        except Exception as exc:
            result = {
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        if self.state.run_id is not None:
            result["run_id"] = self.state.run_id
        result = json_safe(result)
        self.audit.log(
            "tool_result",
            {"tool": tool, "result": result},
        )
        return result

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

        row = self.rescue[
            pd.to_numeric(self.rescue["panel_size"], errors="coerce") == panel_size
        ]

        if row.empty and panel_size == 269:
            row = self.stage5[
                pd.to_numeric(self.stage5["panel_size"], errors="coerce") == 269
            ]

        if row.empty:
            raise ValueError(
                f"No cached trainval result exists for panel size {panel_size}."
            )

        r = row.iloc[0]
        ba = float(r["balanced_accuracy_mean"])
        auc = float(r["roc_auc_mean"])
        pass_ba = ba >= MIN_BA
        pass_auc = auc >= MIN_AUC

        def maybe_float(col: str):
            return float(r[col]) if col in row.index and pd.notna(r[col]) else None

        return {
            "panel_name": str(r["panel_name"]),
            "panel_size": int(r["panel_size"]),
            "balanced_accuracy_mean": ba,
            "balanced_accuracy_sd": maybe_float("balanced_accuracy_sd"),
            "roc_auc_mean": auc,
            "roc_auc_sd": maybe_float("roc_auc_sd"),
            "mcc_mean": maybe_float("mcc_mean"),
            "mcc_sd": maybe_float("mcc_sd"),
            "sensitivity_pd_mean": maybe_float("sensitivity_pd_mean"),
            "specificity_hc_mean": maybe_float("specificity_hc_mean"),
            "f1_macro_mean": maybe_float("f1_macro_mean"),
            "pr_auc_pd_mean": maybe_float("pr_auc_pd_mean"),
            "passes_ba": bool(pass_ba),
            "passes_auc": bool(pass_auc),
            "passes_both": bool(pass_ba and pass_auc),
            "data_scope": str(r["data_scope"]),
            "test_rows_accessed": int(r["test_rows_accessed"]),
        }

    # ------------------------------------------------------------------
    # Six model-facing tools
    # ------------------------------------------------------------------

    def get_locked_objective(self) -> dict[str, Any]:
        # First call starts a run. A call after finalization starts another.
        if self.state.run_id is None or self.state.finalized:
            self.state = AgentState(
                run_id="pending",
                started_at_utc=utc_now(),
            )
            run_id = self.audit.start(self.files)
            self.state.run_id = run_id

        def body():
            self.state.objective_seen = True
            return {
                "status": "OK",
                "goal": (
                    "Select the smallest evidence-guided reduced gene panel "
                    "that preserves trainval predictive performance."
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
                    "test_tool_exposed_to_claude": False,
                },
                "required_next_step": "Inspect external evidence.",
            }

        return self._log_call("get_locked_objective", {}, body)

    def get_external_evidence_summary(self) -> dict[str, Any]:
        blocked = self._active_run_required()
        if blocked:
            return blocked

        def body():
            if not self.state.objective_seen:
                return {
                    "status": "REFUSED",
                    "reason": "Call get_locked_objective first.",
                }

            self.state.evidence_seen = True

            pub = pd.to_numeric(
                self.evidence["pubmed_evidence_flag"], errors="coerce"
            ).fillna(0)
            st = pd.to_numeric(
                self.evidence["string_network_flag"], errors="coerce"
            ).fillna(0)

            group_counts = (
                self.evidence["fast_track_evidence_group"]
                .fillna("UNKNOWN")
                .value_counts()
                .to_dict()
            )

            return {
                "status": "OK",
                "candidate_universe": 593,
                "pubmed_positive_genes": int((pub == 1).sum()),
                "string_connected_genes": int((st == 1).sum()),
                "pubmed_and_string_genes": int(((pub == 1) & (st == 1)).sum()),
                "evidence_group_counts": {
                    str(k): int(v) for k, v in group_counts.items()
                },
                "interpretation_boundary": (
                    "PubMed hits are retrieval signals and STRING edges are "
                    "functional associations; neither is a causal/clinical score."
                ),
                "required_next_step": "Inspect the initial evidence-guided candidate.",
            }

        return self._log_call("get_external_evidence_summary", {}, body)

    def get_initial_evidence_candidate(self) -> dict[str, Any]:
        blocked = self._active_run_required()
        if blocked:
            return blocked

        def body():
            if not self.state.evidence_seen:
                return {
                    "status": "REFUSED",
                    "reason": "Call get_external_evidence_summary first.",
                }

            self.state.initial_candidate_seen = True
            return {
                "status": "OK",
                "panel_name": self.stage5_selected["selected_panel"],
                "panel_size": int(self.stage5_selected["selected_panel_size"]),
                "construction": (
                    "All 41 genes with 5/5 selector support plus all genes "
                    "jointly supported by PubMed and STRING."
                ),
                "stage5_selection_note": self.stage5_selected["selection_reason"],
                "requires_predictive_evaluation": True,
                "required_next_step": "Evaluate the 269-gene candidate.",
            }

        return self._log_call("get_initial_evidence_candidate", {}, body)

    def evaluate_panel(self, panel_size: int) -> dict[str, Any]:
        blocked = self._active_run_required()
        if blocked:
            return blocked

        def body():
            if not self.state.initial_candidate_seen:
                return {
                    "status": "REFUSED",
                    "reason": "Call get_initial_evidence_candidate first.",
                }

            allowed = {269, 279, 289, 299, 309, 319, 593}
            if panel_size not in allowed:
                return {
                    "status": "REFUSED",
                    "reason": f"Panel {panel_size} is not an allowed cached candidate.",
                    "allowed_panel_sizes": sorted(allowed),
                }

            if panel_size in {279, 289, 299, 309, 319} and not self.state.rescue_seen:
                return {
                    "status": "REFUSED",
                    "reason": (
                        "Rescued panels are unavailable until the failed "
                        "269-gene seed has triggered run_feedback_rescue."
                    ),
                }

            result = self._result_for_size(panel_size)
            if result["test_rows_accessed"] != 0:
                raise RuntimeError(
                    "Safety violation: cached result reports test access."
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
                "recommended_action": (
                    "If this is the 269-gene seed and it fails, request feedback rescue."
                    if panel_size == 269 and not result["passes_both"]
                    else (
                        "If this is the smallest passing rescued candidate, finalize it."
                        if result["passes_both"] and panel_size < 593
                        else "Use the locked decision policy."
                    )
                ),
            }

        return self._log_call(
            "evaluate_panel",
            {"panel_size": panel_size},
            body,
        )

    def run_feedback_rescue(self, current_panel_size: int) -> dict[str, Any]:
        blocked = self._active_run_required()
        if blocked:
            return blocked

        def body():
            if current_panel_size != 269:
                return {
                    "status": "REFUSED",
                    "reason": "Feedback rescue is defined only from the 269-gene seed.",
                }
            if 269 not in self.state.evaluated_sizes:
                return {
                    "status": "REFUSED",
                    "reason": "Evaluate panel 269 before requesting rescue.",
                }

            seed = self._result_for_size(269)
            if seed["passes_both"]:
                return {
                    "status": "REFUSED",
                    "reason": "269 already passes; rescue is not justified.",
                }

            self.state.rescue_seen = True

            rows = []
            sizes = sorted(
                int(v)
                for v in pd.to_numeric(
                    self.rescue["panel_size"], errors="coerce"
                ).dropna().unique()
                if int(v) > 269
            )
            for size in sizes:
                r = self._result_for_size(size)
                rows.append(
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

            passing = [r["panel_size"] for r in rows if r["passes_both"]]
            next_candidate = min(passing) if passing else 593

            return {
                "status": "OK",
                "rescue_policy": {
                    "seed_size": 269,
                    "priority": (
                        "Restore excluded genes supported by exactly one "
                        "external source before genes supported by neither; "
                        "within each group use pre-existing reproducibility rank."
                    ),
                    "search": (
                        "Historical deterministic run used coarse search then "
                        "10-gene refinement."
                    ),
                },
                "cached_rescue_results": rows,
                "selection_rule": (
                    "Choose the smallest evaluated rescued panel satisfying "
                    "both locked thresholds."
                ),
                "smallest_passing_candidate_from_cached_search": next_candidate,
                "required_next_step": (
                    f"Explicitly call evaluate_panel({next_candidate}) before finalization."
                ),
            }

        return self._log_call(
            "run_feedback_rescue",
            {"current_panel_size": current_panel_size},
            body,
        )

    def finalize_panel(
        self,
        panel_size: int,
        decision_note: str,
    ) -> dict[str, Any]:
        blocked = self._active_run_required()
        if blocked:
            return blocked

        def body():
            if panel_size != 593 and not self.state.rescue_seen:
                return {
                    "status": "REFUSED",
                    "reason": "Reduced-panel finalization requires feedback rescue.",
                }

            if panel_size not in self.state.evaluated_sizes:
                return {
                    "status": "REFUSED",
                    "reason": (
                        f"Panel {panel_size} has not been explicitly evaluated "
                        "during this Claude run."
                    ),
                }

            passing = []
            for size in sorted(
                int(v)
                for v in pd.to_numeric(
                    self.rescue["panel_size"], errors="coerce"
                ).dropna().unique()
                if int(v) > 269
            ):
                r = self._result_for_size(size)
                if r["passes_both"]:
                    passing.append(size)

            if passing:
                smallest = min(passing)
                if panel_size != smallest:
                    return {
                        "status": "REFUSED",
                        "reason": (
                            f"Panel {panel_size} is not the smallest passing "
                            f"rescued panel. Smallest passing panel is {smallest}."
                        ),
                    }
            elif panel_size != 593:
                return {
                    "status": "REFUSED",
                    "reason": (
                        "No reduced panel passes both constraints; only the "
                        "593-gene reference may be finalized."
                    ),
                }

            requested = self._result_for_size(panel_size)
            if panel_size < 593 and not requested["passes_both"]:
                return {
                    "status": "REFUSED",
                    "reason": "Requested reduced panel fails a locked threshold.",
                }
            if requested["test_rows_accessed"] != 0:
                raise RuntimeError("Safety violation during finalization.")

            self.state.finalized = True
            self.state.final_panel_size = panel_size
            self.state.final_result = requested
            self.state.final_decision_note = decision_note

            result = {
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
                    "Smallest cached rescued panel satisfying both locked "
                    "trainval preservation constraints."
                    if panel_size < 593
                    else "No reduced candidate satisfied both constraints."
                ),
                "decision_note": decision_note,
                "heldout_test_used": False,
                "test_rows_accessed": 0,
            }
            return result

        result = self._log_call(
            "finalize_panel",
            {"panel_size": panel_size, "decision_note": decision_note},
            body,
        )

        if result.get("status") == "FINALIZED":
            self.audit.finalize(self.state)
            result["audit_jsonl"] = str(self.audit.jsonl_path)
            result["audit_markdown"] = str(self.audit.md_path)
            result["summary_json"] = str(self.audit.summary_path)
            # Append paths to tool result audit after files exist.
            self.audit.log(
                "final_artifacts",
                {
                    "audit_jsonl": result["audit_jsonl"],
                    "audit_markdown": result["audit_markdown"],
                    "summary_json": result["summary_json"],
                },
            )

        return result


def project_root_from_env() -> Path:
    value = os.environ.get("MEZBAH_PROJECT_ROOT")
    if not value:
        raise EnvironmentError(
            "MEZBAH_PROJECT_ROOT is not set. In Claude Desktop config, set it "
            "to the absolute path of the Mezbah_Transcriptomics project."
        )
    return Path(value).expanduser().resolve()
