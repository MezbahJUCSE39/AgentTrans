# Pipeline

Every stage writes files that the next stage reads. Nothing is passed in memory between stages, so
any stage can be re-run independently and the agent can be replayed over cached outputs.

## S0 — Consensus feature selection

`feature_selection/01_feature_selection_models.py`

19,884 protein-coding genes, 613 training participants. Five rankers are fit **inside each of five
cross-validation folds** so that no gene is ranked using data it will later be validated on:

| Ranker | Captures |
|---|---|
| Welch *t* | univariate shift, robust to unequal variance and 75/25 class imbalance |
| Mutual information (kNN) | non-linear, non-monotone dependence |
| mRMR | multivariate — maximum relevance, minimum redundancy |
| L1 stability selection | 50 stratified half-subsamples, ranked by selection frequency |
| XGBoost gain | embedded, non-linear, averaged over three seeds |

Top-1000 lists are intersected across rankers:

| Support | Genes | Tier name |
|---|---|---|
| 5 / 5 | 41 | `core_5of5` |
| ≥ 4 / 5 | 185 (144 additional) | `strong_4of5` |
| ≥ 3 / 5 | 593 (408 additional) | `broad_3of5` — the candidate universe **U** |

## S1 — Frozen evidence state

`agentic/tools/statistical_evidence.py` → `agentic/data/gene_evidence_master.csv`

One row per candidate gene: Ensembl ID, symbol, per-method top-100/500/1000 support,
reproducibility rank, individual selector ranks, consensus rank, Welch *t*, log2 fold change,
Cohen's *d*, mean log2 TPM, mutual-information score, L1 selection frequency, tree importance, and
empty columns reserved for external evidence.

Integrity was verified when the table was built (`data/source_validation.json`): 593 genes at
support ≥ 3, tier counts 41 / 144 / 408, zero rank mismatches against the trainval score file, and
exact reproduction of the prior 41- and 185-gene lists by filtering on support.

## S2 — External evidence

**PubMed** (`tools/pubmed_tool.py`) — NCBI E-utilities, cached to `data/pubmed_cache.json`.
A gene counts as PubMed-supported when a curated PD-context query returns at least one article.

| Panel | PubMed-supported |
|---|---|
| 41 | 39.0% |
| 185 | 44.9% |
| 593 | 44.4% |

**STRING** (`tools/string_consensus_comparison.py`) — protein–protein interaction network plus
functional enrichment, evaluated against both a whole-genome background and the 593-gene
background.

| Panel | Observed edges | Expected | PPI enrichment *p* |
|---|---|---|---|
| 41 | 7 | 6 | 0.36 (not enriched) |
| 185 | — | — | 2.2 × 10⁻¹⁶ |
| 593 | 2,399 | 1,473 | < 1 × 10⁻¹⁶ |

Reactome enrichment for the 593 set: *Immune System* FDR 1.2 × 10⁻¹⁹, *Neutrophil degranulation*
FDR 2.5 × 10⁻¹¹. The 41-gene core returns **no** term at FDR < 0.05 — the most reproducible genes
are not the most biologically coherent ones, which is what motivated evidence-guided refinement
rather than simply taking the smallest tier.

## S3 — Hard evidence filter (the one that failed)

`agentic/examples/07_evaluate_evidence_guided_panels.py`

Keep the core 41 plus every gene with **dual** evidence (PubMed *and* STRING) → 269 genes.

| Panel | BA | AUROC | MCC | verdict |
|---|---|---|---|---|
| 593 reference | 0.651 | 0.700 | 0.268 | — |
| 269 seed | 0.596 | 0.662 | 0.175 | **fails both thresholds** |

Pre-registered acceptance thresholds (reference − 0.02): BA ≥ 0.6312, AUROC ≥ 0.6801.

## S4 — Feedback-guided rescue

`agentic/examples/08_run_feedback_guided_rescue.py`,
config `agentic/config/feedback_rescue_config.json`

The rescue order is frozen **before** any new ML result is seen:

1. genes with exactly one external source (PubMed-only or STRING-only) — 225 genes;
2. genes with no external source — 99 genes;
3. within each group, ascending pre-existing reproducibility rank, then gene name.

Search: coarse sizes 319, 369, 419, 469, 519, 569, 593, stopping at the first panel passing both
thresholds, then refine downward in 10-gene steps.

| Panel | BA | AUROC | Passes |
|---|---|---|---|
| 269 | 0.596 | 0.662 | no |
| **279** | **0.635** | **0.690** | **yes** |
| 289 | pass | pass | yes |
| 299 | fail BA | — | no |
| 309 | fail BA | — | no |
| 319 | pass | pass | yes |

Performance is **non-monotone** in panel size — 299 and 309 fail while 279 and 289 pass — which is
precisely why a fixed cut-off would have been the wrong tool and a search was needed.

Ten genes are restored, all from the 4/5 tier: CEP19, SYTL3, FBXL13, BTNL8, F2RL1, NFE4, IFRD1,
LRRC4, WDFY4 (STRING-only) and HS1BP3 (PubMed-only).

## S5 — Agent decision

See the agent section of the main README. The LLM re-derives the S3 → S4 → stop decision through
MCP tools over cached trainval results, and finalization is validated against the minimality rule.

## S6 — Final held-out evaluation

`agentic/examples/11_final_279_evaluation.py` (1,803 lines, run once, by a human)

Refits both the frozen 279-gene panel and the 593-gene reference on the full training set with
inner-CV hyperparameter selection, scores the 154-subject held-out set a single time, and computes
2,000-replicate paired stratified bootstrap intervals on the difference. Outputs land in
`agentic/results/final_279_evaluation/` — metrics, confusion matrices, ROC/PR figures, per-subject
predictions, hyperparameters with convergence assertions, and the bootstrap draws.
