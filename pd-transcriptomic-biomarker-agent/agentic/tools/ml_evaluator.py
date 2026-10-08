"""
Trainval-only ML evaluation tool for agentic gene-panel refinement.

SECURITY / METHODOLOGY BOUNDARY
-------------------------------
This module exposes NO held-out-test evaluation function.

Every candidate panel is evaluated ONLY on rows marked split='trainval'
in splits.csv. The original cv_fold values are used as the outer folds.
Within each outer-training fold, logistic-regression hyperparameters are
tuned in an inner stratified CV.

The intended use is:
    candidate panel -> nested development CV -> agent observes result

The held-out test set must remain inaccessible to the refinement agent.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_MAP = {
    "Healthy Control": 0,
    "Parkinson's Disease": 1,
}


def _make_logistic_pipeline(random_state: int = 42):
    """
    Reproduce the regularized Logistic Regression family used for the
    conventional 593-gene benchmark while remaining compatible across
    scikit-learn versions.
    """
    major_minor = tuple(
        int(v) for v in sklearn.__version__.split(".")[:2]
    )

    if major_minor >= (1, 8):
        clf = LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
            tol=1e-4,
            random_state=random_state,
        )
        grid = {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__l1_ratio": [0.0, 1.0],
        }
    else:
        clf = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )
        grid = {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"],
        }

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    return pipeline, grid


def _load_splits(path: str | Path) -> dict[int, dict[str, Any]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    required = {"PATNO", "label", "split", "cv_fold"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"{path} must contain columns: {sorted(required)}"
        )

    out = {}
    for r in rows:
        patno = int(float(r["PATNO"]))
        if patno in out:
            raise ValueError(f"Duplicate PATNO in split file: {patno}")
        out[patno] = {
            "label": int(float(r["label"])),
            "split": r["split"].strip().lower(),
            "cv_fold": int(float(r["cv_fold"])),
        }
    return out


def _load_expression_trainval(
    expression_csv: str | Path,
    splits_csv: str | Path,
    gene_names: list[str],
):
    expression_csv = Path(expression_csv)
    splits = _load_splits(splits_csv)

    with expression_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        required = {"PATNO", "Visit", "label"}
        if not required.issubset(headers):
            raise ValueError(
                f"{expression_csv} must contain {sorted(required)}"
            )

        missing = [g for g in gene_names if g not in headers]
        if missing:
            raise KeyError(
                f"{len(missing)} requested genes are absent from the "
                f"expression matrix. First missing genes: {missing[:20]}"
            )

        X_rows = []
        y_rows = []
        fold_rows = []
        patnos = []

        seen_patnos = set()

        for r in reader:
            patno = int(float(r["PATNO"]))

            if patno in seen_patnos:
                raise ValueError(
                    f"Duplicate PATNO row in expression matrix: {patno}"
                )
            seen_patnos.add(patno)

            if r["Visit"].strip().upper() != "BL":
                raise ValueError(
                    f"Non-BL visit detected for PATNO {patno}: {r['Visit']}"
                )

            if patno not in splits:
                raise KeyError(f"PATNO {patno} missing from splits.csv")

            meta = splits[patno]

            # HARD BOUNDARY: silently skip the held-out test set.
            if meta["split"] != "trainval":
                continue

            label_text = r["label"].strip()
            if label_text not in LABEL_MAP:
                raise ValueError(
                    f"Unexpected label for PATNO {patno}: {label_text}"
                )
            y = LABEL_MAP[label_text]

            if y != meta["label"]:
                raise ValueError(
                    f"Label mismatch for PATNO {patno}: "
                    f"expression={y}, splits={meta['label']}"
                )

            values = []
            for g in gene_names:
                raw = r[g].strip()
                values.append(float(raw) if raw != "" else np.nan)

            X_rows.append(values)
            y_rows.append(y)
            fold_rows.append(meta["cv_fold"])
            patnos.append(patno)

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=int)
    folds = np.asarray(fold_rows, dtype=int)
    patnos = np.asarray(patnos, dtype=int)

    if len(X) == 0:
        raise ValueError("No trainval samples were loaded.")
    if np.any(folds < 0):
        raise ValueError(
            "Trainval rows must have non-negative cv_fold values."
        )

    return X, y, folds, patnos


def _fold_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
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
        "specificity_hc": float(tn / max(tn + fp, 1)),
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _panel_hash(gene_names: Iterable[str]) -> str:
    canonical = "\n".join(sorted(set(gene_names)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def evaluate_panel_trainval(
    gene_names: Iterable[str],
    expression_csv: str | Path,
    splits_csv: str | Path,
    *,
    random_state: int = 42,
    n_inner_folds: int = 5,
    n_jobs: int = -1,
    cache_json: str | Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate an arbitrary candidate gene panel on TRAINVAL ONLY.

    Returns nested outer-fold performance. Balanced accuracy is the
    hyperparameter-selection metric.

    Parameters
    ----------
    gene_names:
        Iterable of gene symbols matching expression-matrix columns.
    expression_csv:
        BL HC/PD expression matrix with PATNO, Visit, label, and genes.
        For the agentic study this should normally be the 593-gene
        ML-ready expression matrix.
    splits_csv:
        Original splits.csv.
    cache_json:
        Optional persistent JSON cache. Repeated evaluation of an
        identical panel returns the cached result.
    """
    gene_names = list(dict.fromkeys(g.strip() for g in gene_names if g.strip()))
    if not gene_names:
        raise ValueError("Candidate panel is empty.")

    key = _panel_hash(gene_names)

    cache_path = Path(cache_json) if cache_json else None
    cache = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if key in cache:
            return cache[key]

    X, y, outer_folds, patnos = _load_expression_trainval(
        expression_csv, splits_csv, gene_names
    )

    fold_ids = sorted(np.unique(outer_folds).tolist())
    if len(fold_ids) < 2:
        raise ValueError(
            f"Need at least two trainval outer folds, found {fold_ids}"
        )

    fold_results = []

    for outer_fold in fold_ids:
        tr = outer_folds != outer_fold
        va = outer_folds == outer_fold

        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

        inner_cv = StratifiedKFold(
            n_splits=n_inner_folds,
            shuffle=True,
            random_state=random_state + int(outer_fold),
        )

        pipe, grid = _make_logistic_pipeline(random_state)

        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="balanced_accuracy",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            verbose=0,
        )
        search.fit(X_tr, y_tr)

        best = search.best_estimator_
        pred = best.predict(X_va)
        score = best.predict_proba(X_va)[:, 1]

        m = _fold_metrics(y_va, pred, score)
        m.update({
            "outer_fold": int(outer_fold),
            "n_train": int(tr.sum()),
            "n_validation": int(va.sum()),
            "inner_best_balanced_accuracy": float(search.best_score_),
            "best_params": search.best_params_,
        })
        fold_results.append(m)

    metric_names = [
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
    ]

    aggregate = {}
    for metric in metric_names:
        vals = np.asarray([r[metric] for r in fold_results], dtype=float)
        aggregate[f"{metric}_mean"] = float(np.mean(vals))
        aggregate[f"{metric}_sd"] = float(
            np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        )

    result = {
        "data_scope": "trainval_only",
        "test_rows_accessed": 0,
        "panel_hash": key,
        "panel_size": len(gene_names),
        "gene_names": gene_names,
        "n_trainval": int(len(y)),
        "n_hc_trainval": int((y == 0).sum()),
        "n_pd_trainval": int((y == 1).sum()),
        "outer_folds": fold_ids,
        "model_family": "regularized_logistic_regression",
        "selection_metric": "balanced_accuracy",
        **aggregate,
        "fold_results": fold_results,
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache[key] = result
        cache_path.write_text(
            json.dumps(cache, indent=2),
            encoding="utf-8",
        )

    return result
