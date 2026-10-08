from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.statistical_evidence import GeneEvidenceStore

store = GeneEvidenceStore(ROOT / "data" / "gene_evidence_master.csv")

print("Panel sizes:")
print("5/5:", len(store.panel(5)))
print(">=4/5:", len(store.panel(4)))
print(">=3/5:", len(store.panel(3)))

print("\nExample gene:")
print(store.get_gene_evidence("IFITM1"))

print("\nTop 10 by absolute effect size:")
rows = store.panel(3)
rows = sorted(rows, key=lambda r: abs(r["cohens_d"]), reverse=True)[:10]
for r in rows:
    print(r["gene_name"], r["gene_id"], r["cohens_d"], r["tier"])
