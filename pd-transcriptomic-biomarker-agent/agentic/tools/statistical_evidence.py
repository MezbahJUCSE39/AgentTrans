"""
Statistical evidence tool for the PD agentic biomarker project.

This tool is intentionally deterministic: it reads only the evidence table
produced from the user's completed feature-selection analysis. It does not
call an LLM or any external database.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


_NUMERIC_INT = {
    "reproducibility_rank", "support_top100", "support_top500",
    "support_top1000", "in_panel_41", "in_panel_185", "in_panel_593",
    "best_rank", "consensus_rank", "welch_rank", "mutual_info_rank",
    "mrmr_rank", "l1_stability_rank", "tree_rank",
}

_NUMERIC_FLOAT = {
    "mean_rank_present", "median_rank_present", "mean_rank_support",
    "welch_t", "log2FC_PD_vs_HC", "cohens_d", "mean_log2TPM",
    "mutual_info_score", "l1_stability_frequency", "tree_importance",
    "consensus_mean_rank",
}


def _convert(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            out[key] = None
        elif key in _NUMERIC_INT:
            out[key] = int(float(value))
        elif key in _NUMERIC_FLOAT:
            out[key] = float(value)
        else:
            out[key] = value
    return out


class GeneEvidenceStore:
    """Read-only lookup over the 593-gene evidence table."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        with self.csv_path.open(newline="", encoding="utf-8-sig") as f:
            self.rows = [_convert(r) for r in csv.DictReader(f)]

        self.by_id = {str(r["gene_id"]).upper(): r for r in self.rows}
        self.by_name = {str(r["gene_name"]).upper(): r for r in self.rows}

        if len(self.rows) != 593:
            raise ValueError(
                f"Expected 593 candidate genes, found {len(self.rows)}."
            )
        if len(self.by_id) != len(self.rows):
            raise ValueError("Duplicate gene IDs in evidence store.")
        if len(self.by_name) != len(self.rows):
            raise ValueError("Duplicate gene names in evidence store.")

    def get_gene_evidence(self, gene: str) -> dict[str, Any]:
        """
        Look up one gene by Ensembl ID or gene name.
        """
        key = gene.strip().upper()
        row = self.by_id.get(key) or self.by_name.get(key)
        if row is None:
            raise KeyError(f"Gene not in 593-gene candidate universe: {gene}")
        return dict(row)

    def get_many(self, genes: Iterable[str]) -> list[dict[str, Any]]:
        return [self.get_gene_evidence(g) for g in genes]

    def panel(self, min_support: int = 3) -> list[dict[str, Any]]:
        """
        Return the predefined stability tier:
            min_support=5 -> 41 genes
            min_support=4 -> 185 genes
            min_support=3 -> 593 genes
        """
        if min_support not in (3, 4, 5):
            raise ValueError("min_support must be 3, 4, or 5.")
        return [
            dict(r) for r in self.rows
            if int(r["support_top1000"]) >= min_support
        ]

    def panel_gene_names(self, min_support: int = 3) -> list[str]:
        return [r["gene_name"] for r in self.panel(min_support)]

    def panel_gene_ids(self, min_support: int = 3) -> list[str]:
        return [r["gene_id"] for r in self.panel(min_support)]

    def top_by(
        self,
        field: str,
        n: int = 20,
        descending: bool = True,
        min_support: int = 3,
    ) -> list[dict[str, Any]]:
        candidates = [
            r for r in self.panel(min_support)
            if r.get(field) is not None
        ]
        if not candidates:
            raise ValueError(f"No numeric/non-empty values for field: {field}")
        candidates.sort(key=lambda r: r[field], reverse=descending)
        return [dict(r) for r in candidates[:n]]


def get_statistical_evidence(
    gene: str,
    evidence_csv: str | Path = "agentic/data/gene_evidence_master.csv",
) -> dict[str, Any]:
    """
    Convenience function that can later be wrapped directly as an agent tool.
    """
    return GeneEvidenceStore(evidence_csv).get_gene_evidence(gene)
