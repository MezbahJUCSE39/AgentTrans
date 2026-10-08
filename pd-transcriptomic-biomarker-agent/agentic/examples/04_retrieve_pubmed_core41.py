"""Retrieve PubMed evidence for the 41 core (5/5) genes."""
from pathlib import Path
import csv
import sys

AGENTIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTIC))

from tools.pubmed_tool import batch_search_pubmed_pd
from tools.statistical_evidence import GeneEvidenceStore

store = GeneEvidenceStore(AGENTIC / "data" / "gene_evidence_master.csv")
genes = store.panel_gene_names(min_support=5)
assert len(genes) == 41

results = batch_search_pubmed_pd(
    genes,
    cache_path=AGENTIC / "data" / "pubmed_cache.json",
    max_records=10,
)

summary_path = AGENTIC / "data" / "pubmed_core41_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
    fields = [
        "gene_name",
        "total_hit_count",
        "retrieved_record_count",
        "has_error",
        "error_message",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "gene_name": result.get("gene_name", ""),
                "total_hit_count": result.get("total_hit_count", ""),
                "retrieved_record_count": result.get("retrieved_record_count", ""),
                "has_error": 1 if "error" in result else 0,
                "error_message": result.get("message", ""),
            }
        )

print(f"\nSaved: {summary_path}")
print("Full article metadata and abstracts are cached in data/pubmed_cache.json")
