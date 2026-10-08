"""
Example trainval-only evaluation.

Run from the project root after placing this stage1 folder under ./agentic:

python agentic/examples/02_evaluate_panel_trainval.py

Expected local project files:
    splits.csv
    ML_ready_data/593_PD_expression_matrix_BL_HC_PD_labeled.csv
"""
from pathlib import Path
import json
import sys

AGENTIC = Path(__file__).resolve().parents[1]
PROJECT = AGENTIC.parent
sys.path.insert(0, str(AGENTIC))

from tools.statistical_evidence import GeneEvidenceStore
from tools.ml_evaluator import evaluate_panel_trainval

store = GeneEvidenceStore(
    AGENTIC / "data" / "gene_evidence_master.csv"
)

# Start with the fixed 593-gene conventional baseline.
genes = store.panel_gene_names(min_support=3)

result = evaluate_panel_trainval(
    genes,
    expression_csv=PROJECT / "ML_ready_data" / "593_PD_expression_matrix_BL_HC_PD_labeled.csv",
    splits_csv=PROJECT / "splits.csv",
    cache_json=AGENTIC / "runs" / "ml_eval_cache.json",
)

summary = {
    k: v for k, v in result.items()
    if k.endswith("_mean") or k.endswith("_sd")
}
summary["panel_size"] = result["panel_size"]
summary["data_scope"] = result["data_scope"]
summary["test_rows_accessed"] = result["test_rows_accessed"]

print(json.dumps(summary, indent=2))
