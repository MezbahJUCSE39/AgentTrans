"""Small PubMed smoke test before batch retrieval."""
from pathlib import Path
import sys

AGENTIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTIC))

from tools.pubmed_tool import PubMedClient
from tools.statistical_evidence import GeneEvidenceStore

store = GeneEvidenceStore(AGENTIC / "data" / "gene_evidence_master.csv")
example_genes = ["IFITM1", "HECW2", "FPR2", "PROK2", "FCGR2B"]

for gene in example_genes:
    store.get_gene_evidence(gene)

client = PubMedClient(cache_path=AGENTIC / "data" / "pubmed_cache.json")

for gene in example_genes:
    result = client.search_gene_pd(gene, max_records=5)
    print("\n" + "=" * 72)
    print(gene)
    print("Total PubMed hits:", result["total_hit_count"])
    print("Records retrieved:", result["retrieved_record_count"])
    print("Cache hit:", result["cache_hit"])
    for article in result["records"][:3]:
        print(f"- PMID {article['pmid']}: {article['title']}")
