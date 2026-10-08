# Results

All values below are reproduced from files in `agentic/results/`. Each table names its source.

## Cohort

PPMI baseline whole-blood RNA-seq, 767 participants (578 PD / 189 HC, 75.4% prevalence).
Train/validation 613 (462 PD / 151 HC) with five fixed patient-level folds; held-out 154
(116 PD / 38 HC).

## Development, nested cross-validation (trainval only)

Source: `agentic/results/feedback_rescue/feedback_rescue_results.csv`,
`agentic/config/agentic_config.json`

| Panel | Balanced accuracy | AUROC | MCC | Passes locked thresholds |
|---|---|---|---|---|
| 593 (reference) | 0.651 | 0.700 | 0.268 | reference |
| 269 (hard evidence filter) | 0.596 | 0.662 | 0.175 | ✗ both |
| **279 (agent-selected)** | **0.635** | **0.690** | **0.245** | ✓ |
| 289 | pass | pass | — | ✓ |
| 299 | fail | — | — | ✗ BA |
| 309 | fail | — | — | ✗ BA |
| 319 | pass | pass | — | ✓ |

Locked thresholds: BA ≥ 0.6312178588, AUROC ≥ 0.6800514259.

## Held-out test, scored once

Source: `agentic/results/final_279_evaluation/reports/heldout_comparison_279_vs_593.csv`

| Metric | 593 | 279 | Δ (279 − 593) |
|---|---|---|---|
| Accuracy | 0.597 | 0.591 | −0.006 |
| Balanced accuracy | 0.565 | 0.551 | −0.013 |
| Sensitivity (PD) | 0.629 | 0.629 | 0.000 |
| Specificity (HC) | 0.500 | 0.474 | −0.026 |
| MCC | 0.114 | 0.091 | −0.023 |
| AUROC | 0.662 | 0.625 | −0.037 |
| PR-AUC (PD) | 0.875 | 0.857 | −0.017 |
| PR-AUC (HC) | 0.364 | 0.312 | −0.052 |

Note the PR-AUC baseline for the PD class is the prevalence, 0.753 — a PR-AUC of 0.86 is therefore
only modestly above chance and should never be quoted without that baseline.

### Paired bootstrap, 2,000 stratified replicates (seed 20260924)

Source: `reports/paired_bootstrap_difference_279_vs_593.csv`

| Metric | Point Δ | 95% CI |
|---|---|---|
| AUROC | −0.037 | [−0.083, 0.008] |
| Balanced accuracy | −0.013 | [−0.084, 0.061] |
| Accuracy | −0.006 | [−0.058, 0.045] |

Every interval contains zero. Prediction agreement between the two panels is 88.9%;
Spearman correlation between predicted scores is 0.91.

## Evidence density

Source: `agentic/results/feedback_rescue/`, `agentic/data/combined_pubmed_string_593_gene_evidence.csv`

| | Universe U (593) | Final panel P* (279) |
|---|---|---|
| ≥ 1 external source | 82.1% | 97.5% |
| Dual evidence (PubMed **and** STRING) | 40.8% | 86.7% |

Compression therefore more than doubles dual-evidence density while costing 0.013 balanced accuracy
on the held-out set.

## Final model

Source: `reports/final_279_hyperparameters.json`

| | |
|---|---|
| Model | regularized logistic regression (`saga`) |
| `C` | 0.01 |
| `l1_ratio` | 0.0 (ridge) |
| `max_iter` | 10,000 |
| Converged | true, 91 iterations |
| Captured convergence warnings | 0 |
| Inner-CV balanced accuracy | 0.634 |
| `test_rows_accessed` | 0 during selection |

## Agent runs

Source: `agentic/results/claude_mcp_agent/`

| Run ID | Host | Calls | Evaluated panels | Final | Test rows |
|---|---|---|---|---|---|
| `20260924_024304_adf2ec73` | Claude Desktop | 7 | 269, 279 | 279 | 0 |
| `20260924_090331_61ba973f` | Claude Cowork | 7 | 269, 279 | 279 | 0 |
| `20260924_090357_a86d9ec9` | adversarial probes | 18 | 269, 279, 289, 593 | 279 | 0 |

The two independent hosts produced **identical tool trajectories**, which is the reproducibility
claim for the agent layer: given the same frozen state and the same prompt, the controller's
decision path does not drift between hosts.

> **Note on `predictions/heldout_predictions_279_and_593.csv`:** PPMI `PATNO` identifiers were
> replaced with sequential `subject_id` values for public release. Row order is preserved, so the
> paired bootstrap in `bootstrap/` is reproducible from the released file.
