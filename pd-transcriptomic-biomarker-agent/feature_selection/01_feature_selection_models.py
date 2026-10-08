"""
03_feature_selection_models.py
==============================================================================
Leakage-free feature selection for bulk RNA-seq (PPMI, ~19k mRNA genes) and
ML classification on top-K genes (K = 100, 500, 1000, all).

Runs AFTER 1_2_together.py, which writes:
    modeling_table.parquet   (PATNO, Visit, label, label_str, ENSG... columns)
    splits.csv               (PATNO, label, split in {trainval,test}, cv_fold)

------------------------------------------------------------------------------
WHY THIS DESIGN
------------------------------------------------------------------------------
1. NO LEAKAGE. Every step that looks at the data (expression filter, ranking,
   scaling, model fit) is fit on the TRAINING part only:
     * CV phase   : for each of the 5 folds, genes are ranked on the 4 training
                    folds and evaluated on the held-out fold.
     * Test phase : genes are ranked on the full trainval set, models are fit
                    on trainval, and the untouched test set is scored ONCE.
   Selecting genes on all samples and then cross-validating is the classic
   high-dimensional mistake (Ambroise & McLachlan, PNAS 2002) and gives
   optimistic AUCs even on pure noise.

2. EFFICIENT. Rankings are computed ONCE per training set and then sliced for
   every K, so the cost is (#folds + 1) x rankers, not x K x models.
   The expensive multivariate rankers run on a pre-filtered candidate pool.

3. COMPLEMENTARY RANKERS + CONSENSUS
   Stage 0  Expression filter   : drop genes not expressed (log2TPM > MIN_EXPR)
                                  in >= MIN_FRAC of training samples, and
                                  zero-variance genes.
   Stage 1  Candidate pool      : top N_PREFILTER genes by |Welch t| (default)
                                  or by variance.
   Stage 2  Rankers (on pool)   :
      welch        univariate, Welch t-test (robust to unequal variances and
                   class imbalance).
      mutual_info  univariate, non-linear dependence (kNN MI estimator).
      mrmr         multivariate filter: max relevance (F-stat), min redundancy
                   (mean |Pearson r| to already-selected genes). Removes
                   co-expressed near-duplicates, which dominate transcriptomes.
      l1_stability embedded: L1-logistic regression on 50 stratified half-
                   subsamples; genes ranked by selection frequency
                   (Meinshausen & Buhlmann 2010 stability selection).
      tree         embedded, non-linear: XGBoost gain importance (falls back to
                   ExtraTrees if xgboost is not installed), averaged over seeds.
      consensus    mean-rank (Borda) aggregation of the rankers above; usually
                   the most stable list and a good default for the paper.
   Ties (e.g., genes never picked by L1 / trees) are broken by |Welch t|.

4. MODELS: dummy baseline, elastic-net logistic regression, SVM-RBF,
   random forest, XGBoost (HistGradientBoosting fallback).
   Class imbalance is handled with class weights (optional SMOTE for the
   non-tree linear/kernel models only).

5. OUTPUTS (OUT_DIR)
   cv_fold_results.csv     every fold x ranker x K x model
   cv_summary.csv          mean/sd over folds  <- choose your config HERE
   test_results.csv        held-out test metrics (report, don't tune on it)
   stability.csv           Jaccard / Kuncheva index of top-K lists across folds
   gene_scores_trainval.csv per-gene statistics and ranks on full trainval
   top_genes/<ranker>_top<K>.txt   gene lists for enrichment / SHAP
   auroc_vs_k.png          CV AUROC vs K, one panel per model
==============================================================================
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             matthews_corrcoef, f1_score, confusion_matrix)
from sklearn.exceptions import ConvergenceWarning

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as _xgb_err:        # ImportError, or XGBoostError when libomp/libxgboost is missing
    HAS_XGB = False
    print(f"[!] xgboost unavailable ({type(_xgb_err).__name__}): {str(_xgb_err).strip().splitlines()[0]}\n"
          "    -> falling back to ExtraTrees (ranking) and HistGradientBoosting (model).\n"
          "    On macOS/conda the usual fix is: conda install -c conda-forge xgboost libomp")

# ============================ CONFIG — EDIT ME ============================
TABLE          = "modeling_table.parquet"
SPLITS         = "splits.csv"
PATNO_COL      = "PATNO"
LABEL_COL      = "label"
GENE_PREFIX    = "ENSG"
OUT_DIR        = "fs_results"
SEED           = 42
N_JOBS         = -1            # -1 = all cores

# Optional: text file with one ENSG id per line to drop BEFORE selection
# (e.g., chrY genes + XIST if PD/HC sex ratios differ, or hemoglobin genes).
EXCLUDE_GENES_FILE = None

# Stage 0: expression filter (fit on training samples only)
MIN_EXPR       = 1.0           # log2TPM value that counts as "expressed"
MIN_FRAC       = 0.20          # expressed in >= 20% of training samples

# Stage 1: candidate pool for the multivariate rankers
PREFILTER_MODE = "welch"       # "welch" (supervised, train-only) or "variance" (unsupervised)
N_PREFILTER    = 5000          # must be >= max numeric K

# Stage 2: rankers
RANKERS        = ["welch", "mutual_info", "mrmr", "l1_stability", "tree"]
ADD_CONSENSUS  = True
TOP_K          = [100, 500, 1000, "all"]    # "all" = every gene column, no selection

MI_NEIGHBORS       = 3
STAB_N_SUBSAMPLES  = 50        # stability selection subsamples
STAB_FRACTION      = 0.5       # fraction of each class per subsample
STAB_TARGET_NNZ    = 100       # C is auto-tuned so one L1 fit selects ~this many genes
TREE_N_REPEATS     = 3         # seeds averaged for tree importance
MRMR_FLOOR         = 1e-3      # redundancy floor (as in the mrmr_selection package)

# Models
MODELS         = ["dummy", "enet_lr", "svm_rbf", "rf", "xgb"]
USE_SMOTE      = False         # if True (needs imbalanced-learn): SMOTE for enet_lr/svm_rbf only
RUN_CV         = True
RUN_TEST       = True
# =========================================================================

SK_GE_18 = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)
NUMERIC_K = sorted(k for k in TOP_K if k != "all")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_logreg(l1_ratio, **kw):
    """LogisticRegression that works on sklearn <1.8 and >=1.8 ('penalty' deprecated in 1.8)."""
    if SK_GE_18:
        return LogisticRegression(l1_ratio=l1_ratio, **kw)
    penalty = {0.0: "l2", 1.0: "l1"}.get(float(l1_ratio), "elasticnet")
    if penalty == "elasticnet":
        return LogisticRegression(penalty=penalty, l1_ratio=l1_ratio, **kw)
    return LogisticRegression(penalty=penalty, **kw)


# ============================== DATA =====================================
def load_data():
    if TABLE.endswith(".parquet"):
        df = pd.read_parquet(TABLE)
    elif TABLE.endswith(".pkl"):
        df = pd.read_pickle(TABLE)
    else:
        df = pd.read_csv(TABLE)
    splits = pd.read_csv(SPLITS)[[PATNO_COL, "split", "cv_fold"]]
    for d in (df, splits):
        d[PATNO_COL] = pd.to_numeric(d[PATNO_COL], errors="coerce").astype("int64")
    df = df.merge(splits, on=PATNO_COL, how="inner")   # all rows of a patient share its split

    gene_cols = [c for c in df.columns if c.startswith(GENE_PREFIX)]
    if EXCLUDE_GENES_FILE:
        with open(EXCLUDE_GENES_FILE) as fh:
            excl = {line.strip().split(".")[0] for line in fh if line.strip()}
        before = len(gene_cols)
        gene_cols = [g for g in gene_cols if g.split(".")[0] not in excl]
        log(f"excluded {before - len(gene_cols)} genes listed in {EXCLUDE_GENES_FILE}")

    X = df[gene_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=int)
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).any(axis=0).sum())
        sys.exit(f"[!] {n_bad} genes contain NaN/inf. Impute or drop them before feature selection.")

    log(f"data: {X.shape[0]} samples x {X.shape[1]} genes | PD={int(y.sum())} HC={int((y == 0).sum())}")
    log(f"expression range: min={X.min():.2f} median={np.median(X):.2f} max={X.max():.2f} "
        f"(MIN_EXPR={MIN_EXPR} is applied on this scale)")
    return df, X, y, np.array(gene_cols)


# ============================ RANKERS ====================================
def welch_t(X, y):
    X = X.astype(np.float64, copy=False)
    a, b = X[y == 1], X[y == 0]
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    return (a.mean(0) - b.mean(0)) / np.maximum(se, 1e-12)


def order_by(primary, tiebreak):
    """Indices sorted by primary desc, then tiebreak desc."""
    return np.lexsort((-tiebreak, -primary))


def rank_mrmr(Z, relevance, k, tiebreak):
    """FCQ mRMR: relevance / mean |corr| with already-selected genes. Z is column-standardized."""
    n, p = Z.shape
    k = min(k, p)
    selected = np.empty(k, dtype=int)
    available = np.ones(p, dtype=bool)
    red_sum = np.zeros(p)
    for i in range(k):
        if i == 0:
            score = relevance.copy()
        else:
            score = relevance / np.maximum(red_sum / i, MRMR_FLOOR)
        score[~available] = -np.inf
        j = int(np.argmax(score))
        selected[i] = j
        available[j] = False
        red_sum += np.abs(Z.T @ Z[:, j]) / n          # |Pearson r| of gene j with all genes
    rest = np.flatnonzero(available)
    rest = rest[order_by(relevance[rest], tiebreak[rest])]
    return np.concatenate([selected, rest])


def _stratified_subsample(y, frac, rng):
    idx = [rng.choice(np.flatnonzero(y == c), size=max(2, int(frac * (y == c).sum())), replace=False)
           for c in (0, 1)]
    return np.concatenate(idx)


def _l1_fit(Z, y, idx, C, seed):
    clf = make_logreg(1.0, solver="liblinear", C=C, class_weight="balanced",
                      max_iter=5000, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Z[idx], y[idx])
    return clf.coef_.ravel()


def rank_l1_stability(Z, y, tiebreak, seed):
    rng = np.random.default_rng(seed)
    subs = [_stratified_subsample(y, STAB_FRACTION, rng) for _ in range(STAB_N_SUBSAMPLES)]
    # auto-tune C on the first subsample so one fit selects ~STAB_TARGET_NNZ genes
    grid = np.logspace(-3, 1, 17)
    nnz = Parallel(n_jobs=N_JOBS)(delayed(_l1_fit)(Z, y, subs[0], C, seed) for C in grid)
    nnz = np.array([(c != 0).sum() for c in nnz])
    C = grid[int(np.argmin(np.abs(nnz - STAB_TARGET_NNZ)))]
    coefs = np.vstack(Parallel(n_jobs=N_JOBS)(
        delayed(_l1_fit)(Z, y, s, C, seed + b) for b, s in enumerate(subs)))
    freq = (coefs != 0).mean(0)
    mag = np.abs(coefs).mean(0)
    order = np.lexsort((-tiebreak, -mag, -freq))
    info = dict(C=float(C), mean_nnz=float((coefs != 0).sum(1).mean()), n_ever=int((freq > 0).sum()))
    return order, freq, info


def rank_tree(X, y, tiebreak, seed):
    imp = np.zeros(X.shape[1])
    pos, neg = int(y.sum()), int((y == 0).sum())
    for r in range(TREE_N_REPEATS):
        if HAS_XGB:
            m = XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.3, min_child_weight=2, importance_type="gain",
                              scale_pos_weight=neg / max(pos, 1), tree_method="hist",
                              eval_metric="logloss", n_jobs=os.cpu_count(), random_state=seed + r)
        else:
            m = ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced",
                                     min_samples_leaf=2, n_jobs=N_JOBS, random_state=seed + r)
        m.fit(X, y)
        fi = np.asarray(m.feature_importances_, dtype=float)
        imp += fi / fi.sum() if fi.sum() > 0 else fi
    imp /= TREE_N_REPEATS
    return order_by(imp, tiebreak), imp


def rank_genes(Xtr, ytr, seed=SEED, verbose=True):
    """
    Fit ALL selection steps on (Xtr, ytr) only.
    Returns rankings {name: global gene indices best->worst (pool only)} and a per-gene score table.
    """
    t0 = time.time()
    p_all = Xtr.shape[1]

    # ---- Stage 0: expression + variance filter ----
    var = Xtr.var(0)
    keep = ((Xtr > MIN_EXPR).mean(0) >= MIN_FRAC) & (var > 1e-8)
    idx0 = np.flatnonzero(keep)
    t_all = welch_t(Xtr[:, idx0], ytr)

    # ---- Stage 1: candidate pool ----
    pool_score = np.abs(t_all) if PREFILTER_MODE == "welch" else var[idx0]
    sel = np.argsort(-pool_score, kind="stable")[:N_PREFILTER]
    pool = idx0[sel]
    t = t_all[sel]
    abs_t = np.abs(t)
    Xp = Xtr[:, pool].astype(np.float32)
    mu, sd = Xp.mean(0), Xp.std(0)
    Z = (Xp - mu) / np.maximum(sd, 1e-8)
    if verbose:
        log(f"  stage0 expressed genes: {len(idx0)}/{p_all} | stage1 pool: {len(pool)} ({PREFILTER_MODE})")

    local, scores, infos = {}, {}, {}
    if "welch" in RANKERS:
        local["welch"] = np.argsort(-abs_t, kind="stable")
    if "mutual_info" in RANKERS:
        ts = time.time()
        try:
            mi = mutual_info_classif(Xp, ytr, discrete_features=False, n_neighbors=MI_NEIGHBORS,
                                     random_state=seed, n_jobs=N_JOBS)
        except TypeError:   # sklearn < 1.5 has no n_jobs
            mi = mutual_info_classif(Xp, ytr, discrete_features=False, n_neighbors=MI_NEIGHBORS,
                                     random_state=seed)
        local["mutual_info"] = order_by(mi, abs_t)
        scores["mutual_info"] = mi
        infos["mutual_info_sec"] = round(time.time() - ts, 1)
    if "mrmr" in RANKERS:
        ts = time.time()
        local["mrmr"] = rank_mrmr(Z, t ** 2, max(NUMERIC_K), abs_t)
        infos["mrmr_sec"] = round(time.time() - ts, 1)
    if "l1_stability" in RANKERS:
        ts = time.time()
        local["l1_stability"], freq, info = rank_l1_stability(Z, ytr, abs_t, seed)
        scores["l1_stability_freq"] = freq
        infos.update({f"l1_{k}": v for k, v in info.items()})
        infos["l1_sec"] = round(time.time() - ts, 1)
    if "tree" in RANKERS:
        ts = time.time()
        local["tree"], imp = rank_tree(Xp, ytr, abs_t, seed)
        scores["tree_importance"] = imp
        infos["tree_sec"] = round(time.time() - ts, 1)

    # rank position of every pool gene under each ranker
    positions = {}
    for name, order in local.items():
        pos = np.empty(len(pool), dtype=int)
        pos[order] = np.arange(len(pool))
        positions[name] = pos
    if ADD_CONSENSUS and len(local) > 1:
        mean_rank = np.mean([positions[n] for n in local], axis=0)
        local["consensus"] = np.lexsort((-abs_t, mean_rank))
        pos = np.empty(len(pool), dtype=int)
        pos[local["consensus"]] = np.arange(len(pool))
        positions["consensus"] = pos
        scores["consensus_mean_rank"] = mean_rank

    rankings = {name: pool[order] for name, order in local.items()}

    a, b = Xp[ytr == 1], Xp[ytr == 0]
    table = pd.DataFrame({
        "gene_index": pool,
        "welch_t": t,
        "log2FC_PD_vs_HC": a.mean(0) - b.mean(0),
        "cohens_d": (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2 + 1e-12),
        "mean_log2TPM": Xp.mean(0),
    })
    for k, v in scores.items():
        table[k] = v
    for name, pos in positions.items():
        table[f"rank_{name}"] = pos + 1
    if verbose:
        log(f"  ranking done in {time.time() - t0:.1f}s  {infos}")
    return rankings, table, infos


# ============================== MODELS ===================================
def make_model(name, y_train):
    pos, neg = int(y_train.sum()), int((y_train == 0).sum())
    cw = None if USE_SMOTE else "balanced"
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "enet_lr":
        clf = make_logreg(0.5, solver="saga", C=0.1, class_weight=cw, max_iter=3000, tol=1e-3,
                          random_state=SEED)
    elif name == "svm_rbf":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=cw, random_state=SEED)
    elif name == "rf":
        return RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                                      class_weight="balanced_subsample", n_jobs=1, random_state=SEED)
    elif name == "xgb":
        if HAS_XGB:
            return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.5, min_child_weight=2,
                                 scale_pos_weight=neg / max(pos, 1), tree_method="hist",
                                 eval_metric="logloss", n_jobs=1, random_state=SEED)
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                              class_weight="balanced", random_state=SEED)
    else:
        raise ValueError(name)

    if USE_SMOTE:   # only for linear / kernel models, never for trees
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        return ImbPipeline([("scale", StandardScaler()), ("smote", SMOTE(random_state=SEED)), ("clf", clf)])
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def metrics(y_true, score, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        auroc=roc_auc_score(y_true, score),
        auprc=average_precision_score(y_true, score),
        bal_acc=balanced_accuracy_score(y_true, y_pred),
        mcc=matthews_corrcoef(y_true, y_pred),
        f1=f1_score(y_true, y_pred, zero_division=0),
        sensitivity=tp / max(tp + fn, 1),
        specificity=tn / max(tn + fp, 1),
    )


def fit_eval(Xtr, ytr, Xev, yev, gene_idx, model_name):
    t0 = time.time()
    model = make_model(model_name, ytr)
    A, B = Xtr[:, gene_idx], Xev[:, gene_idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(A, ytr)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(B)[:, 1]
    else:
        score = model.decision_function(B)
    out = metrics(yev, score, model.predict(B))
    out["fit_sec"] = round(time.time() - t0, 2)
    return out


def feature_sets(rankings, n_genes):
    sets = []
    for method, order in rankings.items():
        for k in NUMERIC_K:
            sets.append((method, k, order[:k]))
    if "all" in TOP_K:
        sets.append(("no_selection", "all", np.arange(n_genes)))
    return sets


def evaluate(Xtr, ytr, Xev, yev, rankings, extra):
    sets = feature_sets(rankings, Xtr.shape[1])
    jobs = [(m, k, idx, mdl) for (m, k, idx) in sets for mdl in MODELS]
    res = Parallel(n_jobs=N_JOBS)(delayed(fit_eval)(Xtr, ytr, Xev, yev, idx, mdl)
                                  for (_, _, idx, mdl) in jobs)
    rows = []
    for (m, k, idx, mdl), r in zip(jobs, res):
        rows.append({**extra, "ranker": m, "K": k, "n_genes": len(idx), "model": mdl, **r})
    return rows


# ============================= STABILITY =================================
def stability_table(fold_rankings, n_total):
    """Mean pairwise Jaccard and Kuncheva index of top-K sets across CV folds."""
    rows = []
    folds = sorted(fold_rankings)
    for method in fold_rankings[folds[0]]:
        for k in NUMERIC_K:
            sets = [set(fold_rankings[f][method][:k]) for f in folds]
            jac, kun = [], []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    inter = len(sets[i] & sets[j])
                    kk = min(len(sets[i]), len(sets[j]))
                    jac.append(inter / len(sets[i] | sets[j]))
                    kun.append((inter * n_total - kk ** 2) / (kk * (n_total - kk)))
            in_all = len(set.intersection(*sets))
            rows.append(dict(ranker=method, K=k, jaccard_mean=np.mean(jac),
                             kuncheva_mean=np.mean(kun), genes_in_all_folds=in_all))
    return pd.DataFrame(rows)


# =============================== PLOT ====================================
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot_auroc_vs_k(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in MODELS if m != "dummy"]
    present = set(summary["ranker"])
    rankers = [r for r in RANKERS + ["consensus"] if r in present]   # fixed order -> fixed color per ranker
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mdl in zip(axes, models):
        s = summary[summary.model == mdl]
        for i, r in enumerate(rankers):
            d = s[s.ranker == r].copy()
            d = d[d.K != "all"].assign(Kn=lambda x: x.K.astype(int)).sort_values("Kn")
            ax.errorbar(d.Kn, d.auroc_mean, yerr=d.auroc_sd, color=SERIES_COLORS[i % 8],
                        marker=MARKERS[i % 8], ms=6, lw=2, capsize=2, label=r)
        allrow = s[s.ranker == "no_selection"]
        if len(allrow):
            ax.axhline(allrow.auroc_mean.iloc[0], color="#52514e", ls="--", lw=1.2,
                       label="all genes (no selection)")
        ax.set_xscale("log")
        ax.set_xticks(NUMERIC_K)
        ax.set_xticklabels([str(k) for k in NUMERIC_K])
        ax.set_title(mdl, fontsize=11, color="#0b0b0b")
        ax.set_xlabel("Top-K genes", color="#52514e")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("CV AUROC (mean ± sd)", color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =============================== MAIN ====================================
def summarize(df):
    metric_cols = ["auroc", "auprc", "bal_acc", "mcc", "f1", "sensitivity", "specificity"]
    g = df.groupby(["ranker", "K", "n_genes", "model"], sort=False)[metric_cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
    return out.sort_values("auroc_mean", ascending=False)


def main():
    os.makedirs(os.path.join(OUT_DIR, "top_genes"), exist_ok=True)
    assert N_PREFILTER >= max(NUMERIC_K), "N_PREFILTER must be >= the largest K"
    log(f"sklearn {sklearn.__version__} | xgboost available: {HAS_XGB} | rankers: {RANKERS}")
    df, X, y, genes = load_data()
    is_tv = (df["split"] == "trainval").to_numpy()
    is_te = (df["split"] == "test").to_numpy()
    folds = df["cv_fold"].to_numpy()

    with open(os.path.join(OUT_DIR, "run_config.json"), "w") as fh:
        json.dump({k: v for k, v in globals().items() if k.isupper() and
                   isinstance(v, (int, float, str, list, bool, type(None)))}, fh, indent=2)

    # ------------------------- 1. CROSS-VALIDATION -------------------------
    if RUN_CV:
        cv_rows, fold_rankings = [], {}
        fold_ids = sorted(f for f in np.unique(folds[is_tv]) if f >= 0)
        for f in fold_ids:
            tr = is_tv & (folds != f)
            va = is_tv & (folds == f)
            log(f"CV fold {f}: train={tr.sum()} (PD={y[tr].sum()}) | val={va.sum()} (PD={y[va].sum()})")
            rankings, _, infos = rank_genes(X[tr], y[tr], seed=SEED + f)
            fold_rankings[f] = rankings
            cv_rows += evaluate(X[tr], y[tr], X[va], y[va], rankings, {"fold": f})
            pd.DataFrame(cv_rows).to_csv(os.path.join(OUT_DIR, "cv_fold_results.csv"), index=False)

        cv = pd.DataFrame(cv_rows)
        cv["K"] = cv["K"].astype(str)
        summary = summarize(cv)
        summary.to_csv(os.path.join(OUT_DIR, "cv_summary.csv"), index=False)
        stab = stability_table(fold_rankings, X.shape[1])
        stab.to_csv(os.path.join(OUT_DIR, "stability.csv"), index=False)
        plot_auroc_vs_k(summary, os.path.join(OUT_DIR, "auroc_vs_k.png"))

        pd.set_option("display.width", 200)
        log("CV top 15 configurations by AUROC:")
        print(summary[["ranker", "K", "model", "auroc_mean", "auroc_sd", "auprc_mean",
                       "bal_acc_mean", "mcc_mean"]].head(15).round(3).to_string(index=False))
        log("Top-K stability across folds:")
        print(stab.round(3).to_string(index=False))

    # --------------------------- 2. HELD-OUT TEST --------------------------
    if RUN_TEST:
        log(f"FINAL: ranking genes on full trainval ({is_tv.sum()} samples), scoring test ({is_te.sum()})")
        rankings, table, infos = rank_genes(X[is_tv], y[is_tv], seed=SEED)
        table.insert(0, "gene", genes[table["gene_index"].to_numpy()])
        sort_col = "rank_consensus" if "rank_consensus" in table else table.filter(like="rank_").columns[0]
        table.drop(columns="gene_index").sort_values(sort_col).to_csv(
            os.path.join(OUT_DIR, "gene_scores_trainval.csv"), index=False)
        for method, order in rankings.items():
            for k in NUMERIC_K:
                with open(os.path.join(OUT_DIR, "top_genes", f"{method}_top{k}.txt"), "w") as fh:
                    fh.write("\n".join(genes[order[:k]]))

        test = pd.DataFrame(evaluate(X[is_tv], y[is_tv], X[is_te], y[is_te], rankings, {}))
        test["K"] = test["K"].astype(str)
        test = test.sort_values("auroc", ascending=False)
        test.to_csv(os.path.join(OUT_DIR, "test_results.csv"), index=False)

        if RUN_CV:
            best = summary.iloc[0]
            hit = test[(test.ranker == best.ranker) & (test.K == best.K) & (test.model == best.model)].iloc[0]
            log(f"Best config chosen on CV: {best.ranker} | K={best.K} | {best.model} "
                f"-> CV AUROC {best.auroc_mean:.3f}±{best.auroc_sd:.3f} | TEST AUROC {hit.auroc:.3f} "
                f"AUPRC {hit.auprc:.3f} BalAcc {hit.bal_acc:.3f}")
        log(f"done. results in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
