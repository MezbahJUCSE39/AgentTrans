# Obtaining the input data

This repository contains **no subject-level PPMI data**. The two files the pipeline needs must be
rebuilt locally after you obtain PPMI access.

## 1. Request PPMI access

Register and accept the Data Use Agreement at
<https://www.ppmi-info.org/access-data-specimens/download-data>. Approval is typically granted
within a few business days to researchers at recognised institutions.

## 2. Download

| File | Where |
|---|---|
| Whole-blood RNA-seq gene expression matrix (baseline visit, log2 TPM) | PPMI · Omics · Transcriptomics |
| `Participant_Status.csv` | PPMI · Study Data · Subject Characteristics |

## 3. Rebuild the two inputs

`feature_selection/01_feature_selection_models.py` documents the construction. In short:

1. Keep the baseline (`BL`) visit only, one sample per participant.
2. Join expression to `Participant_Status` on `PATNO` and map
   `COHORT_DEFINITION` to a binary label — `Parkinson's Disease` = 1, `Healthy Control` = 0.
   Drop every other cohort (prodromal, SWEDD, genetic registry).
3. Write `modeling_table.parquet`, then a patient-level stratified split with five fixed
   cross-validation folds inside the training portion:

   ```python
   from sklearn.model_selection import StratifiedGroupKFold
   outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
   trainval_idx, test_idx = next(outer.split(pat, pat.label, pat.PATNO))
   ```

4. Save as `splits.csv` with columns `PATNO, label, split, cv_fold`
   (`split` ∈ {`trainval`, `test`}, `cv_fold` ∈ {0..4} for trainval and `-1` for test).
5. Subset the expression matrix to the published gene panels to produce
   `ML_ready_data/593_PD_expression_matrix_BL_HC_PD_labeled.csv`
   (columns: `PATNO, Visit, <gene symbols…>, label`).

## 4. Expected cohort after filtering

| | n | PD | HC |
|---|---|---|---|
| Total baseline, labelled | 767 | 578 | 189 |
| Train / validation | 613 | 462 | 151 |
| Held-out test | 154 | 116 | 38 |

If your counts differ, the downstream cached results will not reconcile — check the cohort filter
and the baseline-visit restriction before continuing.

## 5. Verify

```bash
python agentic/examples/01_inspect_statistical_evidence.py
python agentic/examples/02_evaluate_panel_trainval.py
```

The second script should reproduce the 593-gene reference: balanced accuracy ≈ 0.651,
AUROC ≈ 0.700.
