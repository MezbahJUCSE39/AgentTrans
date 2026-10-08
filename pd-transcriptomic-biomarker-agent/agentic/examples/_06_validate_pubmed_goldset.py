"""
Validate the manually annotated PubMed gold-standard CSV.

Run after you finish human review:

    python agentic/examples/06_validate_pubmed_goldset.py

The script checks:
- no rows remain UNREVIEWED
- evidence_class values are valid
- YES/NO/UNKNOWN fields are valid
- confidence is HIGH/MEDIUM/LOW
- direct evidence is not marked YES for an irrelevant false positive
- independent_study_id is populated
- reason is populated
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


VALID_EVIDENCE_CLASSES = {
    "DIRECT_HUMAN_PD",
    "DIRECT_PD_MODEL",
    "PD_COMPUTATIONAL_BIOMARKER",
    "REVIEW_MECHANISTIC_SUPPORT",
    "INDIRECT_PD_CONTEXT",
    "IRRELEVANT_FALSE_POSITIVE",
}

YN_FIELDS = [
    "direct_gene_evidence",
    "pd_specific",
    "human_evidence",
    "blood_evidence",
    "brain_evidence",
    "experimental_model",
    "biomarker_evidence",
    "mechanistic_evidence",
]

VALID_YNU = {"YES", "NO", "UNKNOWN"}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}


def main():
    ROOT = Path(__file__).resolve().parents[1]
    path = ROOT / "data" / "pubmed_core41_annotation_template.csv"

    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    errors = []

    for i, row in enumerate(rows, start=2):
        aid = row.get("annotation_id", f"row_{i}")

        if row["manual_status"].strip().upper() != "REVIEWED":
            errors.append(f"{aid}: manual_status must be REVIEWED")

        if row["evidence_class"] not in VALID_EVIDENCE_CLASSES:
            errors.append(
                f"{aid}: invalid evidence_class={row['evidence_class']!r}"
            )

        for field in YN_FIELDS:
            if row[field] not in VALID_YNU:
                errors.append(
                    f"{aid}: {field} must be YES/NO/UNKNOWN"
                )

        if row["confidence"] not in VALID_CONFIDENCE:
            errors.append(
                f"{aid}: confidence must be HIGH/MEDIUM/LOW"
            )

        if not row["independent_study_id"].strip():
            errors.append(
                f"{aid}: independent_study_id is empty"
            )

        if not row["reason"].strip():
            errors.append(f"{aid}: reason is empty")

        if (
            row["evidence_class"] == "IRRELEVANT_FALSE_POSITIVE"
            and row["direct_gene_evidence"] == "YES"
        ):
            errors.append(
                f"{aid}: false-positive record cannot have "
                "direct_gene_evidence=YES"
            )

    if errors:
        print(f"Validation FAILED with {len(errors)} issue(s):")
        for e in errors[:100]:
            print("-", e)
        raise SystemExit(1)

    print("=" * 72)
    print("GOLD STANDARD VALIDATION: PASS")
    print("=" * 72)
    print(f"Reviewed rows: {len(rows)}")

    counts = Counter(r["evidence_class"] for r in rows)
    print("\nEvidence classes:")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
