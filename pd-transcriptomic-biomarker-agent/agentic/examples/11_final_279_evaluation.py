#!/usr/bin/env python3
"""
11_final_279_evaluation.py
==========================

PATCHED: explicit target_label handling prevents expression/splits label-name collisions.

Final evaluation of the FROZEN 279-gene feedback-rescue panel.

This script performs, in order:

1. Validate/freeze the 279-gene panel and record SHA256 hashes.
2. Load the 593-gene baseline expression matrix and fixed splits.csv.
3. Re-run convergence-safe nested TRAINVAL evaluation for:
      - 279-gene frozen panel
      - 593-gene reference panel
   using the same Logistic Regression family/grid as the agentic ML evaluator.
4. ABORT before held-out evaluation if the convergence-safe 279-gene
   development result no longer satisfies the LOCKED preservation thresholds.
5. Final hyperparameter selection on ALL trainval rows using the original
   predefined trainval folds and balanced accuracy.
6. Refit one final 279 model and one final 593 reference model on all trainval.
7. Evaluate both ONCE on the fixed held-out test cohort.
8. Run 2,000 stratified bootstrap replicates on fixed held-out predictions.
9. Run paired stratified bootstrap for 279-vs-593 metric differences.
10. Save models, predictions, metrics, confidence intervals, comparisons,
    confusion matrices, HPO tables, and an audit JSON.

IMPORTANT METHODOLOGICAL BOUNDARY
---------------------------------
The 279-gene panel is immutable in this script.

NO feature selection, panel rescue, panel-size search, or threshold changes are
allowed after held-out results are observed.

The held-out cohort has been inspected previously for the conventional
41/185/593 analysis. Therefore, the 279-gene result should be described as a
"secondary internal held-out evaluation", NOT as pristine external validation.

Expected project layout
-----------------------
Mezbah_Transcriptomics/
├── splits.csv
├── ML_ready_data/
│   └── 593_PD_expression_matrix_BL_HC_PD_labeled.csv
└── agentic/
    └── results/
        ├── feedback_rescue/
        │   └── FINAL_FEEDBACK_RESCUE_PANEL.csv
        └── string_consensus_comparison/
            └── combined_pubmed_string_593_gene_evidence.csv

Run
---
From the project root:

    conda activate research

    python agentic/examples/11_final_279_evaluation.py

Optional explicit project root:

    python agentic/examples/11_final_279_evaluation.py \
        --project-root /Users/mezbah/Documents/Mezbah_Transcriptomics

If a completed held-out evaluation flag already exists, the script refuses to
run the held-out stage again. Use --allow-rerun ONLY for a documented technical
rerun; never use it to search for a better result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =====================================================================
# LOCKED STUDY CONSTANTS
# =====================================================================

RANDOM_STATE = 42
BOOTSTRAP_SEED = 20260924
N_BOOTSTRAP = 2000
N_INNER_FOLDS = 5
N_JOBS = -1

# Increased only to address numerical convergence.
# Do not alter feature panel, folds, grid, scoring, class weights, or thresholds.
LR_MAX_ITER = 10000

# Locked development reference established BEFORE feedback rescue.
LOCKED_593_BA = 0.6512178588125292
LOCKED_593_AUC = 0.7000514258999532

# Predeclared maximum loss = 0.02 for BOTH metrics.
MIN_ACCEPTED_BA = 0.6312178588125292
MIN_ACCEPTED_AUC = 0.6800514258999532

EXPECTED_279_SIZE = 279
EXPECTED_593_SIZE = 593
EXPECTED_TRAINVAL_N = 613
EXPECTED_TEST_N = 154

ID_COL = "PATNO"
VISIT_COL = "Visit"
EXPR_LABEL_COL = "label"
BASELINE_VISIT = "BL"

LABEL_MAP = {
    "Healthy Control": 0,
    "Parkinson's Disease": 1,
}

METRIC_ORDER = [
    "accuracy",
    "balanced_accuracy",
    "precision_pd",
    "sensitivity_pd",
    "specificity_hc",
    "f1_pd",
    "f1_macro",
    "mcc",
    "roc_auc",
    "pr_auc_pd",
    "pr_auc_hc",
]


# Historical 593 held-out point estimates from the earlier conventional stage.
# These are NOT used for model selection; they are only used as a consistency
# check against the convergence-safe 593 rerun.
HISTORICAL_593_TEST = {
    "accuracy": 0.597403,
    "balanced_accuracy": 0.564655,
    "precision_pd": 0.793478,
    "sensitivity_pd": 0.629310,
    "specificity_hc": 0.500000,
    "f1_pd": 0.701923,
    "f1_macro": 0.540962,
    "mcc": 0.113675,
    "roc_auc": 0.661751,
    "pr_auc_pd": 0.874776,
    "pr_auc_hc": 0.364496,
}


# =====================================================================
# UTILITIES
# =====================================================================

def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot JSON-serialize {type(obj)}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_gene_order(genes: Iterable[str]) -> str:
    text = "\n".join(str(g) for g in genes)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_path(project_root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def normalize_patno(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="raise").astype(int)


def selected_estimator_converged(estimator: Pipeline) -> dict[str, Any]:
    clf = estimator.named_steps["clf"]
    n_iter = getattr(clf, "n_iter_", None)
    max_iter = getattr(clf, "max_iter", None)

    if n_iter is None or max_iter is None:
        return {
            "check_available": False,
            "converged": None,
            "n_iter": None,
            "max_iter": max_iter,
        }

    arr = np.asarray(n_iter, dtype=float)
    converged = bool(np.max(arr) < float(max_iter))

    return {
        "check_available": True,
        "converged": converged,
        "n_iter": arr.astype(int).tolist(),
        "max_iter": int(max_iter),
    }


# =====================================================================
# MODEL DEFINITION -- MATCHES THE AGENTIC TRAINVAL EVALUATOR
# =====================================================================

def make_logistic_pipeline(random_state: int = RANDOM_STATE):
    """
    Match the regularized Logistic Regression family/grid used in the
    trainval-only agentic evaluator, except max_iter is increased to 10,000
    solely for convergence safety.
    """
    major_minor = tuple(int(v) for v in sklearn.__version__.split(".")[:2])

    if major_minor >= (1, 8):
        clf = LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=LR_MAX_ITER,
            tol=1e-4,
            random_state=random_state,
        )
        grid = {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__l1_ratio": [0.0, 1.0],
        }
        compatibility_mode = "sklearn>=1.8:saga+l1_ratio"
    else:
        clf = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=LR_MAX_ITER,
            random_state=random_state,
        )
        grid = {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"],
        }
        compatibility_mode = "sklearn<1.8:liblinear+penalty"

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    return pipeline, grid, compatibility_mode


# =====================================================================
# DATA LOADING / VALIDATION
# =====================================================================

@dataclass
class DataBundle:
    expression: pd.DataFrame
    splits: pd.DataFrame
    genes_279: list[str]
    genes_593: list[str]
    trainval_patnos: np.ndarray
    test_patnos: np.ndarray
    y_trainval: np.ndarray
    y_test: np.ndarray
    trainval_folds: np.ndarray
    X279_trainval: np.ndarray
    X279_test: np.ndarray
    X593_trainval: np.ndarray
    X593_test: np.ndarray


def load_frozen_279(panel_path: Path) -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_csv(panel_path)

    if "gene_name" not in panel.columns:
        raise ValueError(
            f"{panel_path} must contain a 'gene_name' column."
        )

    if "final_panel_order" in panel.columns:
        panel = panel.sort_values("final_panel_order").reset_index(drop=True)

    genes = panel["gene_name"].astype(str).str.strip().tolist()

    if len(genes) != EXPECTED_279_SIZE:
        raise ValueError(
            f"Frozen panel must contain exactly {EXPECTED_279_SIZE} rows; "
            f"found {len(genes)}."
        )
    if len(set(genes)) != len(genes):
        dup = pd.Series(genes)[pd.Series(genes).duplicated()].tolist()
        raise ValueError(f"Duplicate genes in frozen 279 panel: {dup[:20]}")
    if any(not g for g in genes):
        raise ValueError("Blank gene symbol found in frozen 279 panel.")

    return panel, genes


def load_593_gene_universe(
    evidence_path: Path | None,
    expression: pd.DataFrame,
) -> list[str]:
    if evidence_path is not None and evidence_path.exists():
        ev = pd.read_csv(evidence_path)
        if "gene_name" not in ev.columns:
            raise ValueError(
                f"{evidence_path} exists but lacks 'gene_name'."
            )
        genes = ev["gene_name"].astype(str).str.strip().tolist()
        genes = list(dict.fromkeys(genes))
    else:
        # Fallback only if evidence file is unavailable.
        metadata = {
            "PATNO", "Visit", "label", "label_str",
            "split", "cv_fold", "PD_status",
        }
        genes = [c for c in expression.columns if c not in metadata]

    if len(genes) != EXPECTED_593_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_593_SIZE} genes in the reference universe; "
            f"found {len(genes)}."
        )

    missing = [g for g in genes if g not in expression.columns]
    if missing:
        raise KeyError(
            f"{len(missing)} 593-panel genes are absent from expression matrix. "
            f"First missing: {missing[:20]}"
        )

    return genes


def validate_expression_labels(
    expression: pd.DataFrame,
    merged_splits: pd.DataFrame,
) -> None:
    """
    Verify expression label against splits.csv when an expression-side label
    column is present.

    IMPORTANT:
    Both files may contain a column named ``label``. To avoid pandas creating
    label_x/label_y and making the comparison ambiguous, the splits.csv label
    is renamed to ``target_label`` before merging.

    The authoritative modeling target is always splits.csv['label'].
    """
    if EXPR_LABEL_COL not in expression.columns:
        return

    tmp = expression[[ID_COL, EXPR_LABEL_COL]].copy()
    tmp[ID_COL] = normalize_patno(tmp[ID_COL])

    split_labels = (
        merged_splits[[ID_COL, "label"]]
        .rename(columns={"label": "target_label"})
        .copy()
    )

    m = tmp.merge(
        split_labels,
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )

    # Because the split-side target has already been renamed, the expression
    # column retains its original name ("label") without suffixing.
    raw = m[EXPR_LABEL_COL]

    # Try text labels first.
    text = raw.astype(str).str.strip()
    mapped = text.map(LABEL_MAP)

    if mapped.notna().all():
        expr_y = mapped.astype(int).to_numpy()
    else:
        # Then try numeric labels.
        numeric = pd.to_numeric(raw, errors="coerce")
        if (
            numeric.notna().all()
            and set(numeric.astype(int).unique()).issubset({0, 1})
        ):
            expr_y = numeric.astype(int).to_numpy()
        else:
            examples = text.drop_duplicates().tolist()[:10]
            raise ValueError(
                "Could not interpret expression-matrix label column. "
                f"Examples: {examples}"
            )

    split_y = m["target_label"].astype(int).to_numpy()

    if not np.array_equal(expr_y, split_y):
        bad = m.loc[
            expr_y != split_y,
            [ID_COL, EXPR_LABEL_COL, "target_label"],
        ]
        raise ValueError(
            "Expression labels disagree with splits.csv. "
            f"First mismatches:\n{bad.head(10)}"
        )


def load_data(
    expression_path: Path,
    splits_path: Path,
    panel_path: Path,
    evidence_path: Path | None,
) -> DataBundle:
    expression = pd.read_csv(expression_path)
    splits = pd.read_csv(splits_path)

    required_expr = {ID_COL, VISIT_COL}
    missing_expr = required_expr.difference(expression.columns)
    if missing_expr:
        raise ValueError(
            f"Expression matrix missing columns: {sorted(missing_expr)}"
        )

    required_split = {ID_COL, "label", "split", "cv_fold"}
    missing_split = required_split.difference(splits.columns)
    if missing_split:
        raise ValueError(
            f"splits.csv missing columns: {sorted(missing_split)}"
        )

    expression[ID_COL] = normalize_patno(expression[ID_COL])
    splits[ID_COL] = normalize_patno(splits[ID_COL])

    # Enforce BL only.
    expression[VISIT_COL] = expression[VISIT_COL].astype(str).str.strip()
    non_bl = expression[
        expression[VISIT_COL].str.upper() != BASELINE_VISIT
    ]
    if not non_bl.empty:
        raise ValueError(
            f"Expression matrix contains {len(non_bl)} non-BL rows."
        )

    if expression[ID_COL].duplicated().any():
        dup = expression.loc[
            expression[ID_COL].duplicated(keep=False), ID_COL
        ].tolist()[:20]
        raise ValueError(
            f"Duplicate PATNO rows in expression matrix. Examples: {dup}"
        )

    if splits[ID_COL].duplicated().any():
        dup = splits.loc[
            splits[ID_COL].duplicated(keep=False), ID_COL
        ].tolist()[:20]
        raise ValueError(
            f"Duplicate PATNO rows in splits.csv. Examples: {dup}"
        )

    splits["split"] = splits["split"].astype(str).str.strip().str.lower()
    splits["label"] = pd.to_numeric(splits["label"], errors="raise").astype(int)
    splits["cv_fold"] = pd.to_numeric(
        splits["cv_fold"], errors="raise"
    ).astype(int)

    if not set(splits["label"].unique()).issubset({0, 1}):
        raise ValueError("splits.csv labels must be 0/1.")

    if set(splits["split"].unique()) != {"trainval", "test"}:
        raise ValueError(
            "splits.csv must contain exactly trainval and test split labels. "
            f"Found: {sorted(splits['split'].unique())}"
        )

    # Ensure exactly the same participants exist in expression and split file.
    expr_ids = set(expression[ID_COL])
    split_ids = set(splits[ID_COL])

    missing_expr_ids = sorted(split_ids - expr_ids)
    unexpected_expr_ids = sorted(expr_ids - split_ids)

    if missing_expr_ids or unexpected_expr_ids:
        raise ValueError(
            "Participant mismatch between expression matrix and splits.csv. "
            f"Missing expression PATNOs: {missing_expr_ids[:20]}; "
            f"unexpected expression PATNOs: {unexpected_expr_ids[:20]}"
        )

    validate_expression_labels(expression, splits)

    frozen_panel, genes_279 = load_frozen_279(panel_path)
    genes_593 = load_593_gene_universe(evidence_path, expression)

    not_in_593 = [g for g in genes_279 if g not in set(genes_593)]
    if not_in_593:
        raise ValueError(
            "Frozen 279 panel contains genes outside the 593 reference universe: "
            f"{not_in_593[:20]}"
        )

    missing279 = [g for g in genes_279 if g not in expression.columns]
    if missing279:
        raise KeyError(
            f"{len(missing279)} frozen genes absent from expression matrix: "
            f"{missing279[:20]}"
        )

    # Merge metadata using an explicit authoritative target name.
    # This avoids any dependence on pandas suffix behavior when the expression
    # matrix itself also contains a column named "label".
    split_meta = (
        splits[[ID_COL, "label", "split", "cv_fold"]]
        .rename(columns={"label": "target_label"})
        .copy()
    )

    merged = expression.merge(
        split_meta,
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )

    trainval = merged.loc[merged["split"] == "trainval"].copy()
    test = merged.loc[merged["split"] == "test"].copy()

    if len(trainval) != EXPECTED_TRAINVAL_N:
        raise ValueError(
            f"Expected {EXPECTED_TRAINVAL_N} trainval participants; "
            f"found {len(trainval)}."
        )
    if len(test) != EXPECTED_TEST_N:
        raise ValueError(
            f"Expected {EXPECTED_TEST_N} test participants; found {len(test)}."
        )

    if trainval[ID_COL].duplicated().any() or test[ID_COL].duplicated().any():
        raise ValueError("Duplicate PATNO after split merge.")

    fold_ids = sorted(trainval["cv_fold"].unique().tolist())
    if len(fold_ids) != 5:
        raise ValueError(
            f"Expected five predefined trainval folds; found {fold_ids}"
        )

    # splits.csv is the authoritative source of modeling labels.
    y_trainval = trainval["target_label"].to_numpy(dtype=int)
    y_test = test["target_label"].to_numpy(dtype=int)
    folds = trainval["cv_fold"].to_numpy(dtype=int)

    X279_trainval = trainval[genes_279].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    X279_test = test[genes_279].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    X593_trainval = trainval[genes_593].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    X593_test = test[genes_593].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    if np.isinf(X279_trainval).any() or np.isinf(X279_test).any():
        raise ValueError("Infinite value detected in 279-gene matrix.")
    if np.isinf(X593_trainval).any() or np.isinf(X593_test).any():
        raise ValueError("Infinite value detected in 593-gene matrix.")

    # Missing numeric values are allowed because imputation is in-pipeline.
    return DataBundle(
        expression=expression,
        splits=splits,
        genes_279=genes_279,
        genes_593=genes_593,
        trainval_patnos=trainval[ID_COL].to_numpy(dtype=int),
        test_patnos=test[ID_COL].to_numpy(dtype=int),
        y_trainval=y_trainval,
        y_test=y_test,
        trainval_folds=folds,
        X279_trainval=X279_trainval,
        X279_test=X279_test,
        X593_trainval=X593_trainval,
        X593_test=X593_test,
    )


# =====================================================================
# METRICS
# =====================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision_pd": float(
            precision_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )
        ),
        "sensitivity_pd": float(
            recall_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )
        ),
        "specificity_hc": float(specificity),
        "f1_pd": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc_pd": float(
            average_precision_score(y_true, y_score)
        ),
        "pr_auc_hc": float(
            average_precision_score(1 - y_true, 1 - y_score)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return out


# =====================================================================
# CONVERGENCE-SAFE NESTED TRAINVAL DEVELOPMENT CHECK
# =====================================================================

def nested_trainval_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    outer_folds: np.ndarray,
    panel_name: str,
) -> dict[str, Any]:
    fold_results = []

    fold_ids = sorted(np.unique(outer_folds).tolist())

    for outer_fold in fold_ids:
        tr = outer_folds != outer_fold
        va = outer_folds == outer_fold

        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

        inner_cv = StratifiedKFold(
            n_splits=N_INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + int(outer_fold),
        )

        pipe, grid, compatibility_mode = make_logistic_pipeline()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)

            search = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring="balanced_accuracy",
                cv=inner_cv,
                n_jobs=N_JOBS,
                refit=True,
                verbose=0,
                return_train_score=False,
            )
            search.fit(X_tr, y_tr)

        pred = search.best_estimator_.predict(X_va)
        score = search.best_estimator_.predict_proba(X_va)[:, 1]

        metrics = compute_metrics(y_va, pred, score)
        convergence = selected_estimator_converged(
            search.best_estimator_
        )

        fold_results.append({
            "outer_fold": int(outer_fold),
            "n_train": int(tr.sum()),
            "n_validation": int(va.sum()),
            "inner_best_balanced_accuracy": float(search.best_score_),
            "best_params": search.best_params_,
            "selected_estimator_convergence": convergence,
            "captured_convergence_warning_count": int(
                sum(
                    issubclass(w.category, ConvergenceWarning)
                    for w in caught
                )
            ),
            **metrics,
        })

    aggregate = {}
    for metric in METRIC_ORDER:
        vals = np.asarray(
            [r[metric] for r in fold_results],
            dtype=float,
        )
        aggregate[f"{metric}_mean"] = float(np.mean(vals))
        aggregate[f"{metric}_sd"] = float(
            np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        )

    selected_all_converged = all(
        r["selected_estimator_convergence"]["converged"] is not False
        for r in fold_results
    )

    return {
        "panel_name": panel_name,
        "panel_size": int(X.shape[1]),
        "data_scope": "trainval_only",
        "test_rows_accessed": 0,
        "model_family": "regularized_logistic_regression",
        "selection_metric": "balanced_accuracy",
        "sklearn_compatibility_mode": compatibility_mode,
        "lr_max_iter": LR_MAX_ITER,
        "selected_estimators_converged_all_outer_folds": bool(
            selected_all_converged
        ),
        **aggregate,
        "fold_results": fold_results,
    }


# =====================================================================
# FINAL HPO ON ALL TRAINVAL WITH PREDEFINED FOLDS
# =====================================================================

def predefined_split_from_original_folds(
    folds: np.ndarray,
) -> tuple[PredefinedSplit, dict[int, int]]:
    unique = sorted(np.unique(folds).tolist())
    mapping = {original: i for i, original in enumerate(unique)}
    test_fold = np.asarray(
        [mapping[int(v)] for v in folds],
        dtype=int,
    )
    return PredefinedSplit(test_fold=test_fold), mapping


def final_hpo_and_fit(
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    folds: np.ndarray,
    panel_name: str,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    cv, fold_mapping = predefined_split_from_original_folds(folds)

    pipe, grid, compatibility_mode = make_logistic_pipeline()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)

        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=N_JOBS,
            refit=True,  # refits best pipeline on ALL trainval
            verbose=0,
            return_train_score=True,
        )
        search.fit(X_trainval, y_trainval)

    best = search.best_estimator_
    convergence = selected_estimator_converged(best)

    # Save a compact HPO table.
    cv_df = pd.DataFrame(search.cv_results_)
    keep_cols = [
        c for c in [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
            "params",
        ]
        if c in cv_df.columns
    ]
    cv_df = cv_df[keep_cols].sort_values("rank_test_score")

    info = {
        "panel_name": panel_name,
        "panel_size": int(X_trainval.shape[1]),
        "data_scope": "trainval_only",
        "test_rows_accessed": 0,
        "selection_metric": "balanced_accuracy",
        "cv_scheme": "original_predefined_trainval_folds",
        "original_to_predefined_fold_mapping": {
            str(k): int(v) for k, v in fold_mapping.items()
        },
        "best_params": search.best_params_,
        "best_cv_balanced_accuracy": float(search.best_score_),
        "sklearn_compatibility_mode": compatibility_mode,
        "lr_max_iter": LR_MAX_ITER,
        "selected_estimator_convergence": convergence,
        "captured_convergence_warning_count": int(
            sum(
                issubclass(w.category, ConvergenceWarning)
                for w in caught
            )
        ),
    }

    return best, info, cv_df


# =====================================================================
# BOOTSTRAP
# =====================================================================

def stratified_bootstrap_indices(
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    b0 = rng.choice(idx0, size=len(idx0), replace=True)
    b1 = rng.choice(idx1, size=len(idx1), replace=True)

    return np.concatenate([b0, b1])


def bootstrap_fixed_predictions(
    y: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []

    for b in range(1, n_bootstrap + 1):
        idx = stratified_bootstrap_indices(y, rng)
        m = compute_metrics(y[idx], pred[idx], score[idx])
        rows.append({
            "bootstrap_id": b,
            **{k: m[k] for k in METRIC_ORDER},
        })

    raw = pd.DataFrame(rows)

    ci_rows = []
    for metric in METRIC_ORDER:
        vals = raw[metric].to_numpy(dtype=float)
        ci_rows.append({
            "metric": metric,
            "bootstrap_mean": float(np.mean(vals)),
            "ci_2_5": float(np.percentile(vals, 2.5)),
            "ci_97_5": float(np.percentile(vals, 97.5)),
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(seed),
            "bootstrap_type": "stratified_fixed_prediction",
        })

    return raw, pd.DataFrame(ci_rows)


def paired_bootstrap_difference(
    y: np.ndarray,
    pred279: np.ndarray,
    score279: np.ndarray,
    pred593: np.ndarray,
    score593: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Same resampled participants are used for BOTH models in every replicate.
    Difference convention: 279 - 593.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for b in range(1, n_bootstrap + 1):
        idx = stratified_bootstrap_indices(y, rng)

        m279 = compute_metrics(
            y[idx], pred279[idx], score279[idx]
        )
        m593 = compute_metrics(
            y[idx], pred593[idx], score593[idx]
        )

        row = {"bootstrap_id": b}
        for metric in METRIC_ORDER:
            row[f"{metric}_279"] = m279[metric]
            row[f"{metric}_593"] = m593[metric]
            row[f"delta_{metric}_279_minus_593"] = (
                m279[metric] - m593[metric]
            )
        rows.append(row)

    raw = pd.DataFrame(rows)

    ci_rows = []
    for metric in METRIC_ORDER:
        col = f"delta_{metric}_279_minus_593"
        vals = raw[col].to_numpy(dtype=float)
        ci_rows.append({
            "metric": metric,
            "difference": "279_minus_593",
            "bootstrap_mean_difference": float(np.mean(vals)),
            "ci_2_5": float(np.percentile(vals, 2.5)),
            "ci_97_5": float(np.percentile(vals, 97.5)),
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(seed),
            "bootstrap_type": "paired_stratified_fixed_prediction",
        })

    return raw, pd.DataFrame(ci_rows)


# =====================================================================
# FIGURES
# =====================================================================

def save_confusion_matrix(
    y: np.ndarray,
    pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y,
        pred,
        labels=[0, 1],
        display_labels=["HC", "PD"],
        values_format="d",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_roc_comparison(
    y: np.ndarray,
    score279: np.ndarray,
    score593: np.ndarray,
    path: Path,
) -> None:
    fpr279, tpr279, _ = roc_curve(y, score279)
    fpr593, tpr593, _ = roc_curve(y, score593)

    auc279 = roc_auc_score(y, score279)
    auc593 = roc_auc_score(y, score593)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr279, tpr279, label=f"279 genes (AUC={auc279:.3f})")
    ax.plot(fpr593, tpr593, label=f"593 genes (AUC={auc593:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out ROC comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pr_comparison(
    y: np.ndarray,
    score279: np.ndarray,
    score593: np.ndarray,
    path: Path,
) -> None:
    p279, r279, _ = precision_recall_curve(y, score279)
    p593, r593, _ = precision_recall_curve(y, score593)

    ap279 = average_precision_score(y, score279)
    ap593 = average_precision_score(y, score593)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(r279, p279, label=f"279 genes (AP={ap279:.3f})")
    ax.plot(r593, p593, label=f"593 genes (AP={ap593:.3f})")
    ax.set_xlabel("Recall (PD)")
    ax.set_ylabel("Precision (PD)")
    ax.set_title("Held-out PD precision-recall comparison")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# OUTPUT HELPERS
# =====================================================================

def metrics_to_table(
    metrics279: dict[str, Any],
    metrics593: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for metric in METRIC_ORDER:
        rows.append({
            "metric": metric,
            "panel_593": metrics593[metric],
            "panel_279": metrics279[metric],
            "delta_279_minus_593": (
                metrics279[metric] - metrics593[metric]
            ),
        })
    return pd.DataFrame(rows)


def confusion_to_df(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        index=["Actual_HC", "Actual_PD"],
        columns=["Predicted_HC", "Predicted_PD"],
    )


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Final frozen 279-gene held-out evaluation."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Mezbah_Transcriptomics project root. Default: current directory.",
    )
    parser.add_argument(
        "--expression",
        default="ML_ready_data/593_PD_expression_matrix_BL_HC_PD_labeled.csv",
        help="593-gene BL expression matrix relative to project root.",
    )
    parser.add_argument(
        "--splits",
        default="splits.csv",
        help="Fixed splits.csv relative to project root.",
    )
    parser.add_argument(
        "--panel-279",
        default="agentic/results/feedback_rescue/FINAL_FEEDBACK_RESCUE_PANEL.csv",
        help="Frozen 279-gene panel relative to project root.",
    )
    parser.add_argument(
        "--evidence-593",
        default=(
            "agentic/results/string_consensus_comparison/"
            "combined_pubmed_string_593_gene_evidence.csv"
        ),
        help=(
            "593-gene evidence table used to define reference gene universe. "
            "If absent, the script falls back to expression columns."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="agentic/results/final_279_evaluation",
        help="Output directory relative to project root.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help=f"Number of stratified bootstrap replicates. Default: {N_BOOTSTRAP}.",
    )
    parser.add_argument(
        "--allow-rerun",
        action="store_true",
        help=(
            "Allow held-out stage to rerun if completion flag exists. "
            "Use only for a documented technical rerun, never result searching."
        ),
    )

    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    expression_path = resolve_path(project_root, args.expression)
    splits_path = resolve_path(project_root, args.splits)
    panel279_path = resolve_path(project_root, args.panel_279)
    evidence593_path = resolve_path(project_root, args.evidence_593)
    output_dir = resolve_path(project_root, args.output_dir)

    reports_dir = output_dir / "reports"
    models_dir = output_dir / "models"
    predictions_dir = output_dir / "predictions"
    bootstrap_dir = output_dir / "bootstrap"
    figures_dir = output_dir / "figures"

    for d in [
        output_dir,
        reports_dir,
        models_dir,
        predictions_dir,
        bootstrap_dir,
        figures_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    completion_flag = output_dir / "TEST_EVALUATION_COMPLETE.flag"

    print("=" * 88)
    print("FINAL FROZEN 279-GENE EVALUATION")
    print("=" * 88)
    print(f"Project root: {project_root}")
    print(f"Expression:   {expression_path}")
    print(f"Splits:       {splits_path}")
    print(f"Frozen panel: {panel279_path}")
    print(f"Output:       {output_dir}")
    print()
    print("IMPORTANT: No feature-panel changes are permitted in this script.")
    print()

    # -------------------------------------------------------------
    # 1. Validate files and load data.
    # -------------------------------------------------------------
    for p, label in [
        (expression_path, "expression matrix"),
        (splits_path, "splits.csv"),
        (panel279_path, "frozen 279 panel"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p}")

    evidence_arg = evidence593_path if evidence593_path.exists() else None

    data = load_data(
        expression_path=expression_path,
        splits_path=splits_path,
        panel_path=panel279_path,
        evidence_path=evidence_arg,
    )

    # Copy frozen panel into final-results folder before any test evaluation.
    frozen_copy = reports_dir / "final_279_gene_panel.csv"
    shutil.copy2(panel279_path, frozen_copy)

    panel_file_hash = sha256_file(panel279_path)
    panel_gene_order_hash = sha256_gene_order(data.genes_279)

    print("[PASS] Data validation")
    print(f"       Trainval: {len(data.y_trainval)}")
    print(f"       Test:     {len(data.y_test)}")
    print(f"       HC/PD trainval: {(data.y_trainval==0).sum()}/{(data.y_trainval==1).sum()}")
    print(f"       HC/PD test:     {(data.y_test==0).sum()}/{(data.y_test==1).sum()}")
    print(f"       Frozen genes:   {len(data.genes_279)}")
    print(f"       Reference genes:{len(data.genes_593)}")
    print(f"       279 SHA256:     {panel_file_hash}")
    print()

    # -------------------------------------------------------------
    # 2. Convergence-safe development check. TEST NOT USED.
    # -------------------------------------------------------------
    print("=" * 88)
    print("STEP 1: CONVERGENCE-SAFE NESTED TRAINVAL CHECK")
    print("Held-out rows accessed: 0")
    print("=" * 88)

    dev279 = nested_trainval_evaluation(
        data.X279_trainval,
        data.y_trainval,
        data.trainval_folds,
        "frozen_279",
    )

    dev593 = nested_trainval_evaluation(
        data.X593_trainval,
        data.y_trainval,
        data.trainval_folds,
        "reference_593",
    )

    write_json(
        reports_dir / "development_nested_279.json",
        dev279,
    )
    write_json(
        reports_dir / "development_nested_593.json",
        dev593,
    )

    development_compare = pd.DataFrame([
        {
            "panel": "593",
            "genes": 593,
            "balanced_accuracy_mean": dev593["balanced_accuracy_mean"],
            "balanced_accuracy_sd": dev593["balanced_accuracy_sd"],
            "roc_auc_mean": dev593["roc_auc_mean"],
            "roc_auc_sd": dev593["roc_auc_sd"],
            "mcc_mean": dev593["mcc_mean"],
            "mcc_sd": dev593["mcc_sd"],
            "selected_estimators_converged_all_outer_folds":
                dev593["selected_estimators_converged_all_outer_folds"],
        },
        {
            "panel": "279",
            "genes": 279,
            "balanced_accuracy_mean": dev279["balanced_accuracy_mean"],
            "balanced_accuracy_sd": dev279["balanced_accuracy_sd"],
            "roc_auc_mean": dev279["roc_auc_mean"],
            "roc_auc_sd": dev279["roc_auc_sd"],
            "mcc_mean": dev279["mcc_mean"],
            "mcc_sd": dev279["mcc_sd"],
            "selected_estimators_converged_all_outer_folds":
                dev279["selected_estimators_converged_all_outer_folds"],
        },
    ])

    development_compare["delta_ba_vs_locked_593"] = (
        development_compare["balanced_accuracy_mean"] - LOCKED_593_BA
    )
    development_compare["delta_auc_vs_locked_593"] = (
        development_compare["roc_auc_mean"] - LOCKED_593_AUC
    )

    development_compare.to_csv(
        reports_dir / "development_279_vs_593.csv",
        index=False,
    )

    print(
        f"279 convergence-safe nested BA:  "
        f"{dev279['balanced_accuracy_mean']:.6f} ± "
        f"{dev279['balanced_accuracy_sd']:.6f}"
    )
    print(
        f"279 convergence-safe nested AUC: "
        f"{dev279['roc_auc_mean']:.6f} ± "
        f"{dev279['roc_auc_sd']:.6f}"
    )
    print(
        "279 selected estimators converged in all outer folds: "
        f"{dev279['selected_estimators_converged_all_outer_folds']}"
    )
    print()
    print(
        f"Locked pass thresholds: BA >= {MIN_ACCEPTED_BA:.6f}, "
        f"AUC >= {MIN_ACCEPTED_AUC:.6f}"
    )

    dev279_passes = (
        dev279["balanced_accuracy_mean"] >= MIN_ACCEPTED_BA
        and dev279["roc_auc_mean"] >= MIN_ACCEPTED_AUC
    )

    if not dev279_passes:
        abort_payload = {
            "status": "ABORTED_BEFORE_HELDOUT",
            "reason": (
                "After convergence-safe rerun, the frozen 279-gene panel "
                "no longer satisfies both locked development thresholds."
            ),
            "development_279": {
                "balanced_accuracy_mean": dev279["balanced_accuracy_mean"],
                "roc_auc_mean": dev279["roc_auc_mean"],
            },
            "thresholds": {
                "balanced_accuracy_min": MIN_ACCEPTED_BA,
                "roc_auc_min": MIN_ACCEPTED_AUC,
            },
            "test_rows_accessed": 0,
        }
        write_json(
            output_dir / "ABORTED_BEFORE_HELDOUT.json",
            abort_payload,
        )
        raise RuntimeError(
            "279 failed the locked convergence-safe development thresholds. "
            "Held-out evaluation was NOT performed. "
            "Inspect ABORTED_BEFORE_HELDOUT.json."
        )

    if not dev279["selected_estimators_converged_all_outer_folds"]:
        abort_payload = {
            "status": "ABORTED_BEFORE_HELDOUT",
            "reason": (
                "At least one selected outer-fold estimator did not converge "
                "under max_iter=10000."
            ),
            "test_rows_accessed": 0,
        }
        write_json(
            output_dir / "ABORTED_BEFORE_HELDOUT.json",
            abort_payload,
        )
        raise RuntimeError(
            "Selected 279 estimator failed convergence in at least one "
            "outer fold. Held-out evaluation was NOT performed."
        )

    print("[PASS] Frozen 279 panel still satisfies both locked thresholds.")
    print()

    # -------------------------------------------------------------
    # 3. Final HPO on ALL TRAINVAL, still before held-out use.
    # -------------------------------------------------------------
    print("=" * 88)
    print("STEP 2: FINAL HPO USING TRAINVAL ONLY")
    print("Held-out rows accessed: 0")
    print("=" * 88)

    model279, hpo279, hpo279_df = final_hpo_and_fit(
        data.X279_trainval,
        data.y_trainval,
        data.trainval_folds,
        "frozen_279",
    )
    model593, hpo593, hpo593_df = final_hpo_and_fit(
        data.X593_trainval,
        data.y_trainval,
        data.trainval_folds,
        "reference_593",
    )

    if hpo279["selected_estimator_convergence"]["converged"] is False:
        raise RuntimeError(
            "Final selected 279 model did not converge on all trainval. "
            "Held-out evaluation NOT performed."
        )
    if hpo593["selected_estimator_convergence"]["converged"] is False:
        raise RuntimeError(
            "Final selected 593 reference model did not converge on all trainval. "
            "Held-out evaluation NOT performed."
        )

    write_json(
        reports_dir / "final_279_hyperparameters.json",
        hpo279,
    )
    write_json(
        reports_dir / "final_593_hyperparameters.json",
        hpo593,
    )
    hpo279_df.to_csv(
        reports_dir / "final_279_hpo_grid.csv",
        index=False,
    )
    hpo593_df.to_csv(
        reports_dir / "final_593_hpo_grid.csv",
        index=False,
    )

    joblib.dump(
        model279,
        models_dir / "final_279_model.joblib",
    )
    joblib.dump(
        model593,
        models_dir / "final_593_reference_model.joblib",
    )

    print(
        "279 final params:",
        json.dumps(hpo279["best_params"], default=json_default),
    )
    print(
        f"279 predefined-fold CV BA: "
        f"{hpo279['best_cv_balanced_accuracy']:.6f}"
    )
    print(
        "279 final fit convergence:",
        hpo279["selected_estimator_convergence"],
    )
    print()
    print(
        "593 final params:",
        json.dumps(hpo593["best_params"], default=json_default),
    )
    print(
        f"593 predefined-fold CV BA: "
        f"{hpo593['best_cv_balanced_accuracy']:.6f}"
    )
    print(
        "593 final fit convergence:",
        hpo593["selected_estimator_convergence"],
    )
    print()

    # -------------------------------------------------------------
    # Safety barrier immediately before held-out evaluation.
    # -------------------------------------------------------------
    if completion_flag.exists() and not args.allow_rerun:
        raise RuntimeError(
            f"Held-out completion flag already exists:\n{completion_flag}\n"
            "Refusing to evaluate the test set again. If this is a documented "
            "technical rerun, explicitly pass --allow-rerun."
        )

    print("=" * 88)
    print("STEP 3: ONE HELD-OUT EVALUATION")
    print("=" * 88)
    print(
        "The frozen panel and final model settings are now fixed. "
        "No downstream result will alter them."
    )
    print()

    # -------------------------------------------------------------
    # 4. HELD-OUT EVALUATION. No adaptation after this point.
    # -------------------------------------------------------------
    pred279 = model279.predict(data.X279_test)
    score279 = model279.predict_proba(data.X279_test)[:, 1]

    pred593 = model593.predict(data.X593_test)
    score593 = model593.predict_proba(data.X593_test)[:, 1]

    test279 = compute_metrics(
        data.y_test,
        pred279,
        score279,
    )
    test593 = compute_metrics(
        data.y_test,
        pred593,
        score593,
    )

    # Save participant-level predictions immediately.
    pred_df = pd.DataFrame({
        "PATNO": data.test_patnos,
        "true_label": data.y_test,
        "predicted_279": pred279,
        "probability_PD_279": score279,
        "predicted_593": pred593,
        "probability_PD_593": score593,
    })
    pred_df.to_csv(
        predictions_dir / "heldout_predictions_279_and_593.csv",
        index=False,
    )

    write_json(
        reports_dir / "final_279_holdout_metrics.json",
        test279,
    )
    write_json(
        reports_dir / "final_593_holdout_metrics_recomputed.json",
        test593,
    )

    confusion_to_df(test279).to_csv(
        reports_dir / "final_279_confusion_matrix.csv",
    )
    confusion_to_df(test593).to_csv(
        reports_dir / "final_593_confusion_matrix_recomputed.csv",
    )

    comparison = metrics_to_table(test279, test593)
    comparison.to_csv(
        reports_dir / "heldout_comparison_279_vs_593.csv",
        index=False,
    )

    # Historical 593 consistency check.
    historical_rows = []
    for metric in METRIC_ORDER:
        historical = HISTORICAL_593_TEST[metric]
        recomputed = test593[metric]
        historical_rows.append({
            "metric": metric,
            "historical_593": historical,
            "recomputed_593": recomputed,
            "difference_recomputed_minus_historical": recomputed - historical,
        })
    pd.DataFrame(historical_rows).to_csv(
        reports_dir / "historical_593_consistency_check.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # 5. 2,000 stratified fixed-prediction bootstraps.
    # -------------------------------------------------------------
    print("=" * 88)
    print(f"STEP 4: STRATIFIED BOOTSTRAP ({args.bootstrap} REPLICATES)")
    print("=" * 88)

    raw279, ci279 = bootstrap_fixed_predictions(
        data.y_test,
        pred279,
        score279,
        n_bootstrap=args.bootstrap,
        seed=BOOTSTRAP_SEED,
    )

    raw593, ci593 = bootstrap_fixed_predictions(
        data.y_test,
        pred593,
        score593,
        n_bootstrap=args.bootstrap,
        seed=BOOTSTRAP_SEED,
    )

    paired_raw, paired_ci = paired_bootstrap_difference(
        data.y_test,
        pred279,
        score279,
        pred593,
        score593,
        n_bootstrap=args.bootstrap,
        seed=BOOTSTRAP_SEED,
    )

    raw279.to_csv(
        bootstrap_dir / "bootstrap_raw_279.csv",
        index=False,
    )
    raw593.to_csv(
        bootstrap_dir / "bootstrap_raw_593.csv",
        index=False,
    )
    paired_raw.to_csv(
        bootstrap_dir / "paired_bootstrap_raw_279_vs_593.csv",
        index=False,
    )

    # Add point estimates to CI tables.
    ci279["point_estimate"] = ci279["metric"].map(test279)
    ci593["point_estimate"] = ci593["metric"].map(test593)

    point_delta = {
        m: test279[m] - test593[m]
        for m in METRIC_ORDER
    }
    paired_ci["point_difference"] = paired_ci["metric"].map(
        point_delta
    )

    ci279.to_csv(
        reports_dir / "final_279_bootstrap_ci.csv",
        index=False,
    )
    ci593.to_csv(
        reports_dir / "final_593_bootstrap_ci_recomputed.csv",
        index=False,
    )
    paired_ci.to_csv(
        reports_dir / "paired_bootstrap_difference_279_vs_593.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # 6. Figures.
    # -------------------------------------------------------------
    save_confusion_matrix(
        data.y_test,
        pred279,
        "Frozen 279-gene panel — held-out confusion matrix",
        figures_dir / "279_confusion_matrix.png",
    )
    save_confusion_matrix(
        data.y_test,
        pred593,
        "593-gene reference — held-out confusion matrix",
        figures_dir / "593_confusion_matrix.png",
    )
    save_roc_comparison(
        data.y_test,
        score279,
        score593,
        figures_dir / "heldout_roc_279_vs_593.png",
    )
    save_pr_comparison(
        data.y_test,
        score279,
        score593,
        figures_dir / "heldout_pr_pd_279_vs_593.png",
    )

    # -------------------------------------------------------------
    # 7. Final audit.
    # -------------------------------------------------------------
    audit = {
        "status": "COMPLETE",
        "analysis_stage": "final_frozen_279_secondary_internal_heldout_evaluation",
        "project_root": str(project_root),
        "inputs": {
            "expression_path": str(expression_path),
            "expression_sha256": sha256_file(expression_path),
            "splits_path": str(splits_path),
            "splits_sha256": sha256_file(splits_path),
            "frozen_279_panel_path": str(panel279_path),
            "frozen_279_panel_file_sha256": panel_file_hash,
            "frozen_279_gene_order_sha256": panel_gene_order_hash,
            "evidence_593_path": (
                str(evidence593_path)
                if evidence593_path.exists()
                else None
            ),
        },
        "frozen_panel": {
            "panel_size": 279,
            "panel_modified_after_agent_finalization": False,
            "genes_removed_after_agent_finalization": 0,
            "genes_added_after_agent_finalization": 0,
        },
        "cohort": {
            "trainval_n": int(len(data.y_trainval)),
            "trainval_hc": int((data.y_trainval == 0).sum()),
            "trainval_pd": int((data.y_trainval == 1).sum()),
            "test_n": int(len(data.y_test)),
            "test_hc": int((data.y_test == 0).sum()),
            "test_pd": int((data.y_test == 1).sum()),
        },
        "locked_development_reference": {
            "balanced_accuracy": LOCKED_593_BA,
            "roc_auc": LOCKED_593_AUC,
            "minimum_accepted_balanced_accuracy": MIN_ACCEPTED_BA,
            "minimum_accepted_roc_auc": MIN_ACCEPTED_AUC,
        },
        "convergence_check": {
            "lr_max_iter": LR_MAX_ITER,
            "development_279_passed_locked_thresholds": bool(dev279_passes),
            "development_279_selected_estimators_converged":
                dev279["selected_estimators_converged_all_outer_folds"],
            "development_593_selected_estimators_converged":
                dev593["selected_estimators_converged_all_outer_folds"],
        },
        "final_hpo": {
            "selection_scope": "trainval_only",
            "heldout_used_for_hyperparameter_selection": False,
            "279": hpo279,
            "593": hpo593,
        },
        "heldout": {
            "evaluation_count_in_this_script_run": 1,
            "used_for_feature_selection": False,
            "used_for_hyperparameter_selection": False,
            "panel_changed_after_heldout": False,
            "279_metrics": test279,
            "593_metrics_recomputed": test593,
        },
        "bootstrap": {
            "replicates": int(args.bootstrap),
            "seed": BOOTSTRAP_SEED,
            "individual_ci_type": "stratified_fixed_prediction_percentile",
            "paired_difference_ci_type":
                "paired_stratified_fixed_prediction_percentile",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "interpretation_boundary": (
            "The same held-out cohort was inspected previously during the "
            "conventional 41/185/593 analysis. Therefore this 279-gene "
            "evaluation is a secondary internal held-out assessment, not "
            "pristine external validation."
        ),
    }

    write_json(
        output_dir / "final_evaluation_audit.json",
        audit,
    )

    # Completion flag is written last, only after every major output succeeds.
    completion_flag.write_text(
        "Final frozen 279-gene held-out evaluation completed successfully.\n"
        "Do not modify the gene panel based on these held-out results.\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # 8. Concise terminal summary.
    # -------------------------------------------------------------
    print()
    print("=" * 88)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 88)
    print("Frozen 279 genes: NO post-test modification permitted.")
    print()

    print("279 held-out:")
    print(f"  Accuracy:          {test279['accuracy']:.6f}")
    print(f"  Balanced accuracy: {test279['balanced_accuracy']:.6f}")
    print(f"  Sensitivity PD:    {test279['sensitivity_pd']:.6f}")
    print(f"  Specificity HC:    {test279['specificity_hc']:.6f}")
    print(f"  MCC:               {test279['mcc']:.6f}")
    print(f"  ROC-AUC:           {test279['roc_auc']:.6f}")
    print(f"  PR-AUC PD:         {test279['pr_auc_pd']:.6f}")
    print(f"  PR-AUC HC:         {test279['pr_auc_hc']:.6f}")
    print(
        f"  CM: TN={test279['tn']} FP={test279['fp']} "
        f"FN={test279['fn']} TP={test279['tp']}"
    )
    print()

    print("593 recomputed held-out:")
    print(f"  Balanced accuracy: {test593['balanced_accuracy']:.6f}")
    print(f"  MCC:               {test593['mcc']:.6f}")
    print(f"  ROC-AUC:           {test593['roc_auc']:.6f}")
    print()

    print("279 - 593:")
    print(
        f"  Δ Balanced accuracy: "
        f"{test279['balanced_accuracy'] - test593['balanced_accuracy']:+.6f}"
    )
    print(
        f"  Δ ROC-AUC:           "
        f"{test279['roc_auc'] - test593['roc_auc']:+.6f}"
    )
    print(
        f"  Δ MCC:               "
        f"{test279['mcc'] - test593['mcc']:+.6f}"
    )
    print()
    print(f"Outputs: {output_dir}")
    print(f"Audit:   {output_dir / 'final_evaluation_audit.json'}")
    print()
    print(
        "Methodological label for paper: "
        "'secondary internal held-out evaluation'."
    )


if __name__ == "__main__":
    main()
