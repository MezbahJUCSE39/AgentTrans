"""
Adaptive trainval-only feedback rescue after external-evidence pruning failed.

Starting point:
    269 genes = all 41 core genes + all PubMed+STRING dual-supported genes.

Rescue order:
    1) excluded genes supported by exactly one external source
       (PubMed-only or STRING-only)
    2) excluded genes supported by neither source
    3) within each group: ascending pre-existing reproducibility_rank

Search:
    coarse sizes: 319, 369, 419, 469, 519, 569, 593
    once a coarse panel first passes BOTH locked thresholds, refine between
    the previous failing size and passing size in +10-gene increments.

Locked thresholds:
    BA   >= 0.6312178588125292
    AUC  >= 0.6800514258999532

Held-out test access is forbidden.

Run from project root:
    python agentic/examples/08_run_feedback_guided_rescue.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pandas as pd

AGENTIC = Path(__file__).resolve().parents[1]
PROJECT = AGENTIC.parent
sys.path.insert(0, str(AGENTIC))

from tools.ml_evaluator import evaluate_panel_trainval


REFERENCE_BA = 0.6512178588125292
REFERENCE_AUC = 0.7000514258999532
MIN_BA = 0.6312178588125292
MIN_AUC = 0.6800514258999532

SEED_SIZE = 269
COARSE_SIZES = [319, 369, 419, 469, 519, 569, 593]
FINE_INCREMENT = 10


def read_seed(path: Path) -> list[str]:
    df = pd.read_csv(path)
    genes = df["gene_name"].astype(str).tolist()
    if len(genes) != len(set(genes)):
        raise ValueError("Duplicate genes in seed panel.")
    if len(genes) != SEED_SIZE:
        raise ValueError(f"Expected {SEED_SIZE} seed genes; found {len(genes)}")
    return genes


def read_rescue_order(path: Path) -> list[str]:
    df = pd.read_csv(path)
    genes = df.sort_values("rescue_order")["gene_name"].astype(str).tolist()
    if len(genes) != len(set(genes)):
        raise ValueError("Duplicate genes in rescue order.")
    if len(genes) != 324:
        raise ValueError(f"Expected 324 rescue genes; found {len(genes)}")
    return genes


def passes(result: dict) -> bool:
    return (
        result["balanced_accuracy_mean"] >= MIN_BA
        and result["roc_auc_mean"] >= MIN_AUC
    )


def result_row(panel_name: str, result: dict) -> dict:
    return {
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
        "delta_ba_vs_593": result["balanced_accuracy_mean"] - REFERENCE_BA,
        "delta_auc_vs_593": result["roc_auc_mean"] - REFERENCE_AUC,
        "passes_ba": int(result["balanced_accuracy_mean"] >= MIN_BA),
        "passes_auc": int(result["roc_auc_mean"] >= MIN_AUC),
        "passes_both": int(passes(result)),
        "data_scope": result["data_scope"],
        "test_rows_accessed": result["test_rows_accessed"],
    }


def evaluate_size(
    size: int,
    seed: list[str],
    rescue: list[str],
    expression: Path,
    splits: Path,
    cache: Path,
    outdir: Path,
):
    if size < SEED_SIZE or size > 593:
        raise ValueError(f"Invalid panel size {size}")

    n_add = size - SEED_SIZE
    genes = seed + rescue[:n_add]

    if len(genes) != size or len(set(genes)) != size:
        raise RuntimeError(f"Panel construction failed for n={size}")

    print(f"\nEvaluating rescue panel n={size} (+{n_add} genes) ...")

    result = evaluate_panel_trainval(
        genes,
        expression_csv=expression,
        splits_csv=splits,
        cache_json=cache,
    )

    row = result_row(f"feedback_rescue_{size}", result)

    (outdir / f"feedback_rescue_{size}_full_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(
        f"BA={row['balanced_accuracy_mean']:.4f} "
        f"AUC={row['roc_auc_mean']:.4f} "
        f"MCC={row['mcc_mean']:.4f} "
        f"PASS={bool(row['passes_both'])}"
    )
    return row, genes


def main():
    outdir = AGENTIC / "results" / "feedback_rescue"
    outdir.mkdir(parents=True, exist_ok=True)

    seed = read_seed(
        AGENTIC / "data" / "FINAL_EVIDENCE_GUIDED_PANEL.csv"
    )
    rescue = read_rescue_order(
        AGENTIC / "data" / "feedback_rescue_order.csv"
    )

    expression = (
        PROJECT
        / "ML_ready_data"
        / "593_PD_expression_matrix_BL_HC_PD_labeled.csv"
    )
    splits = PROJECT / "splits.csv"
    cache = outdir / "ml_eval_cache.json"

    print("=" * 80)
    print("FEEDBACK-GUIDED TRAINVAL RESCUE")
    print("=" * 80)
    print(f"Seed: n={SEED_SIZE}")
    print(f"Locked 593 BA:  {REFERENCE_BA:.6f}")
    print(f"Locked 593 AUC: {REFERENCE_AUC:.6f}")
    print(f"Pass BA >=      {MIN_BA:.6f}")
    print(f"Pass AUC >=     {MIN_AUC:.6f}")
    print("Held-out test access: FORBIDDEN")

    evaluated = {}
    gene_lists = {}

    # Seed result may already exist from Stage 5; record it without recomputing.
    seed_row = {
        "panel_name": "feedback_rescue_269_seed",
        "panel_size": 269,
        "balanced_accuracy_mean": 0.5962739597942963,
        "balanced_accuracy_sd": 0.09879821186373253,
        "roc_auc_mean": 0.6622744273024779,
        "roc_auc_sd": 0.09556761281589432,
        "mcc_mean": 0.1748610094337299,
        "mcc_sd": 0.1802271510549055,
        "sensitivity_pd_mean": 0.6901823281907433,
        "specificity_hc_mean": 0.5023655913978494,
        "f1_macro_mean": 0.5773223271514022,
        "pr_auc_pd_mean": 0.8585717149834624,
        "delta_ba_vs_593": 0.5962739597942963 - REFERENCE_BA,
        "delta_auc_vs_593": 0.6622744273024779 - REFERENCE_AUC,
        "passes_ba": 0,
        "passes_auc": 0,
        "passes_both": 0,
        "data_scope": "trainval_only",
        "test_rows_accessed": 0,
    }
    evaluated[269] = seed_row
    gene_lists[269] = seed

    first_passing_coarse = None
    previous_coarse = 269

    # Coarse search.
    for size in COARSE_SIZES:
        row, genes = evaluate_size(
            size, seed, rescue, expression, splits, cache, outdir
        )
        evaluated[size] = row
        gene_lists[size] = genes

        if row["passes_both"]:
            first_passing_coarse = size
            break

        previous_coarse = size

    # Fine search only if a panel below/at 593 first passes.
    if first_passing_coarse is not None:
        lower = previous_coarse
        upper = first_passing_coarse

        fine_sizes = list(
            range(lower + FINE_INCREMENT, upper, FINE_INCREMENT)
        )
        if upper not in fine_sizes:
            fine_sizes.append(upper)

        for size in fine_sizes:
            if size in evaluated:
                continue
            row, genes = evaluate_size(
                size, seed, rescue, expression, splits, cache, outdir
            )
            evaluated[size] = row
            gene_lists[size] = genes

    # Smallest passing tested panel.
    passing_sizes = sorted(
        size for size, row in evaluated.items()
        if row["passes_both"] == 1
    )

    if passing_sizes:
        selected_size = passing_sizes[0]
        selected = evaluated[selected_size]
        reason = (
            "Smallest tested feedback-rescue panel satisfying both "
            "predeclared trainval thresholds."
        )
    else:
        # This should only happen if even reproduced 593 fails, which is a
        # methodological consistency problem.
        selected_size = max(evaluated)
        selected = evaluated[selected_size]
        reason = (
            "No tested panel satisfied both thresholds; investigate evaluator "
            "consistency before any held-out evaluation."
        )

    # Save ordered results.
    ordered_rows = [evaluated[k] for k in sorted(evaluated)]
    result_path = outdir / "feedback_rescue_results.csv"
    with result_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ordered_rows[0].keys()))
        w.writeheader()
        w.writerows(ordered_rows)

    # Build final panel CSV from original evidence table for metadata retention.
    evidence = pd.read_csv(
        AGENTIC / "data" / "combined_pubmed_string_593_gene_evidence.csv"
    )
    selected_genes = gene_lists[selected_size]
    order_map = {g: i + 1 for i, g in enumerate(selected_genes)}
    final = evidence[evidence["gene_name"].isin(selected_genes)].copy()
    final["final_panel_order"] = final["gene_name"].map(order_map)
    final = final.sort_values("final_panel_order")
    final_path = outdir / "FINAL_FEEDBACK_RESCUE_PANEL.csv"
    final.to_csv(final_path, index=False, encoding="utf-8-sig")

    decision = {
        "selected_panel_size": selected_size,
        "selected_panel_name": selected["panel_name"],
        "selection_reason": reason,
        "locked_reference_593": {
            "balanced_accuracy_mean": REFERENCE_BA,
            "roc_auc_mean": REFERENCE_AUC,
        },
        "thresholds": {
            "balanced_accuracy_min": MIN_BA,
            "roc_auc_min": MIN_AUC,
        },
        "selected_result": selected,
        "evaluated_sizes": sorted(evaluated),
        "heldout_test_used_for_selection": False,
    }
    (outdir / "selected_feedback_rescue_panel.json").write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("FEEDBACK RESCUE COMPLETE")
    print("=" * 80)
    print(json.dumps(decision, indent=2))
    print(f"\nResults: {result_path}")
    print(f"Final panel: {final_path}")


if __name__ == "__main__":
    main()
