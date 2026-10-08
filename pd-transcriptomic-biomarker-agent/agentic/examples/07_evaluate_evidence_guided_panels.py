"""
Evaluate evidence-guided candidate panels using the EXISTING trainval-only
nested-CV evaluator.

Run from the project root:

    python agentic/examples/11_evaluate_evidence_guided_panels.py

Expected project files:
    splits.csv
    ML_ready_data/593_PD_expression_matrix_BL_HC_PD_labeled.csv
    agentic/tools/ml_evaluator.py
    agentic/data/evidence_dual_top50.csv
    ...
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

AGENTIC = Path(__file__).resolve().parents[1]
PROJECT = AGENTIC.parent
sys.path.insert(0, str(AGENTIC))

from tools.ml_evaluator import evaluate_panel_trainval


PANELS = [
    "evidence_dual_top50",
    "evidence_dual_top100",
    "evidence_dual_top150",
    "evidence_dual_top200",
    "evidence_dual_all242",
    "core41_plus_dual269",
]

REFERENCE_BA = 0.6512178588125292
REFERENCE_AUC = 0.7000514258999532
BA_TOL = 0.02
AUC_TOL = 0.02

MIN_BA = REFERENCE_BA - BA_TOL
MIN_AUC = REFERENCE_AUC - AUC_TOL


def read_genes(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    genes = [r["gene_name"].strip() for r in rows if r["gene_name"].strip()]
    if len(genes) != len(set(genes)):
        raise ValueError(f"Duplicate genes in {path}")
    return genes


def main():
    outdir = AGENTIC / "results" / "evidence_refinement"
    outdir.mkdir(parents=True, exist_ok=True)

    expression = (
        PROJECT
        / "ML_ready_data"
        / "593_PD_expression_matrix_BL_HC_PD_labeled.csv"
    )
    splits = PROJECT / "splits.csv"
    cache = outdir / "ml_eval_cache.json"

    rows = []

    print("=" * 80)
    print("EVIDENCE-GUIDED TRAINVAL REFINEMENT")
    print("=" * 80)
    print(f"Locked 593 BA:   {REFERENCE_BA:.6f}")
    print(f"Locked 593 AUC:  {REFERENCE_AUC:.6f}")
    print(f"Accept BA >=     {MIN_BA:.6f}")
    print(f"Accept AUC >=    {MIN_AUC:.6f}")
    print("Held-out test access: FORBIDDEN")
    print()

    for panel_name in PANELS:
        genes = read_genes(
            AGENTIC / "data" / f"{panel_name}.csv"
        )

        print(f"\nEvaluating {panel_name} (n={len(genes)}) ...")

        result = evaluate_panel_trainval(
            genes,
            expression_csv=expression,
            splits_csv=splits,
            cache_json=cache,
        )

        row = {
            "panel_name": panel_name,
            "panel_size": result["panel_size"],
            "balanced_accuracy_mean": result["balanced_accuracy_mean"],
            "balanced_accuracy_sd": result["balanced_accuracy_sd"],
            "roc_auc_mean": result["roc_auc_mean"],
            "roc_auc_sd": result["roc_auc_sd"],
            "mcc_mean": result["mcc_mean"],
            "mcc_sd": result["mcc_sd"],
            "sensitivity_pd_mean": result["sensitivity_pd_mean"],
            "specificity_hc_mean": result["specificity_hc_mean"],
            "f1_macro_mean": result["f1_macro_mean"],
            "pr_auc_pd_mean": result["pr_auc_pd_mean"],
            "delta_ba_vs_593": (
                result["balanced_accuracy_mean"] - REFERENCE_BA
            ),
            "delta_auc_vs_593": (
                result["roc_auc_mean"] - REFERENCE_AUC
            ),
            "passes_ba_tolerance": int(
                result["balanced_accuracy_mean"] >= MIN_BA
            ),
            "passes_auc_tolerance": int(
                result["roc_auc_mean"] >= MIN_AUC
            ),
            "passes_both": int(
                result["balanced_accuracy_mean"] >= MIN_BA
                and result["roc_auc_mean"] >= MIN_AUC
            ),
            "data_scope": result["data_scope"],
            "test_rows_accessed": result["test_rows_accessed"],
        }
        rows.append(row)

        print(
            f"BA={row['balanced_accuracy_mean']:.4f} "
            f"AUC={row['roc_auc_mean']:.4f} "
            f"MCC={row['mcc_mean']:.4f} "
            f"PASS={bool(row['passes_both'])}"
        )

        # Preserve full nested-CV output for audit.
        (outdir / f"{panel_name}_full_result.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    # Save summary CSV.
    summary_path = outdir / "evidence_guided_panel_results.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Predeclared selection rule.
    passing = [r for r in rows if r["passes_both"] == 1]

    if passing:
        selected = sorted(
            passing,
            key=lambda r: (
                r["panel_size"],
                -r["balanced_accuracy_mean"],
                -r["roc_auc_mean"],
            ),
        )[0]
        selection_reason = (
            "Smallest candidate satisfying both predeclared "
            "BA and AUROC tolerances."
        )
    else:
        selected = sorted(
            rows,
            key=lambda r: (
                -r["balanced_accuracy_mean"],
                -r["roc_auc_mean"],
                r["panel_size"],
            ),
        )[0]
        selection_reason = (
            "No candidate satisfied both tolerances; selected highest "
            "development balanced accuracy, then AUROC, then smaller size."
        )

    decision = {
        "selected_panel": selected["panel_name"],
        "selected_panel_size": selected["panel_size"],
        "selection_reason": selection_reason,
        "reference_593_balanced_accuracy": REFERENCE_BA,
        "reference_593_roc_auc": REFERENCE_AUC,
        "minimum_accepted_balanced_accuracy": MIN_BA,
        "minimum_accepted_roc_auc": MIN_AUC,
        "selected_result": selected,
        "heldout_test_used_for_selection": False,
    }

    (outdir / "selected_evidence_panel.json").write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )

    # Copy selected gene list into a stable filename.
    selected_src = (
        AGENTIC
        / "data"
        / f"{selected['panel_name']}.csv"
    )
    selected_dst = outdir / "FINAL_EVIDENCE_GUIDED_PANEL.csv"
    selected_dst.write_bytes(selected_src.read_bytes())

    print("\n" + "=" * 80)
    print("DEVELOPMENT-ONLY SELECTION COMPLETE")
    print("=" * 80)
    print(json.dumps(decision, indent=2))
    print(f"\nSummary: {summary_path}")
    print(f"Final panel: {selected_dst}")


if __name__ == "__main__":
    main()
