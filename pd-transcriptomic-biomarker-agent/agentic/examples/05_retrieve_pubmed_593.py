# agentic/examples/05_retrieve_pubmed_593.py

from pathlib import Path
import csv
import sys

AGENTIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTIC))

from tools.pubmed_tool import batch_search_pubmed_pd
from tools.statistical_evidence import GeneEvidenceStore


store = GeneEvidenceStore(
    AGENTIC / "data" / "gene_evidence_master.csv"
)

# >=3/5 selector support = 593 genes
genes = store.panel_gene_names(min_support=3)

assert len(genes) == 593

results = batch_search_pubmed_pd(
    genes,
    cache_path=AGENTIC / "data" / "pubmed_cache.json",
    max_records=10,
)

summary_path = AGENTIC / "data" / "pubmed_593_summary.csv"

with summary_path.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    fields = [
        "gene_name",
        "total_hit_count",
        "retrieved_record_count",
        "pubmed_positive",
        "has_error",
        "error_message",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    for result in results:

        hit_count = result.get(
            "total_hit_count",
            ""
        )

        writer.writerow({
            "gene_name":
                result.get("gene_name", ""),

            "total_hit_count":
                hit_count,

            "retrieved_record_count":
                result.get(
                    "retrieved_record_count",
                    ""
                ),

            "pubmed_positive":
                (
                    1
                    if isinstance(hit_count, int)
                    and hit_count > 0
                    else 0
                ),

            "has_error":
                1 if "error" in result else 0,

            "error_message":
                result.get("message", ""),
        })

print()
print("PubMed retrieval complete.")
print(f"Genes queried: {len(genes)}")
print(f"Saved: {summary_path}")