"""
Comparative STRING analysis for the 41 / 185 / 593 consensus gene panels.

Required inputs
---------------
agentic/data/gene_evidence_master.csv
agentic/data/pubmed_593_summary.csv

Scientific design
-----------------
1. The same 593-gene candidate universe is used for all analyses.
2. Panels are defined only by prior feature-selection stability:
       41  = support 5/5
       185 = support >=4/5
       593 = support >=3/5
3. PubMed hit count is used as a coarse automated literature signal only.
   Retrieval errors are preserved as errors, not treated as zero evidence.
4. STRING is run for ALL three panels with identical settings:
       species = Homo sapiens (9606)
       network_type = functional
       required_score = 400
       add_nodes = 0
5. Cross-panel functional enrichment uses the STRING default human
   background for all three panels.
6. Additional candidate-universe enrichment tests:
       41 vs mapped 593 background
       185 vs mapped 593 background
   This asks what biological terms distinguish stricter consensus panels
   from the broader selected-gene universe.
7. 593 vs 593 is NOT tested, because a panel cannot be enriched against
   itself.

Outputs are designed for a one-day fast-track analysis and are not evidence
of causality or clinical utility.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_BASE = "https://string-db.org/api"
SPECIES = 9606
REQUIRED_SCORE = 400
CALLER_IDENTITY = "mezbah_pd_agentic_bibm_fasttrack"


class StringClient:
    def __init__(self, min_interval: float = 1.05, retries: int = 3):
        self.min_interval = float(min_interval)
        self.retries = int(retries)
        self._last = 0.0

    def _wait(self):
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def request(self, fmt: str, endpoint: str, params: dict, *, binary=False):
        url = f"{API_BASE}/{fmt}/{endpoint}"
        payload = dict(params)
        payload["caller_identity"] = CALLER_IDENTITY

        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                self._wait()
                r = requests.post(url, data=payload, timeout=120)
                self._last = time.monotonic()
                r.raise_for_status()
                return r.content if binary else r.json()
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(2.0 * attempt)

        raise RuntimeError(
            f"STRING request failed after {self.retries} attempts: "
            f"{endpoint}: {last_exc}"
        )

    def map_identifiers(self, genes: list[str], chunk_size: int = 150):
        rows = []
        for start in range(0, len(genes), chunk_size):
            chunk = genes[start:start + chunk_size]
            data = self.request(
                "json",
                "get_string_ids",
                {
                    "identifiers": "\r".join(chunk),
                    "species": SPECIES,
                    "echo_query": 1,
                },
            )
            rows.extend(data)

        df = pd.DataFrame(rows)
        if not df.empty and "queryItem" in df.columns:
            # Keep STRING's first/best result per submitted identifier.
            df = df.drop_duplicates("queryItem", keep="first")
        return df

    def network(self, string_ids: list[str]):
        return self.request(
            "json",
            "network",
            {
                "identifiers": "\r".join(string_ids),
                "species": SPECIES,
                "required_score": REQUIRED_SCORE,
                "network_type": "functional",
                "add_nodes": 0,
            },
        )

    def ppi_enrichment(self, string_ids: list[str]):
        return self.request(
            "json",
            "ppi_enrichment",
            {
                "identifiers": "\r".join(string_ids),
                "species": SPECIES,
                "required_score": REQUIRED_SCORE,
            },
        )

    def enrichment(
        self,
        string_ids: list[str],
        background_ids: list[str] | None = None,
    ):
        params = {
            "identifiers": "\r".join(string_ids),
            "species": SPECIES,
        }
        if background_ids:
            params["background_string_identifiers"] = "\r".join(background_ids)
        return self.request("json", "enrichment", params)

    def network_image(self, string_ids: list[str]):
        return self.request(
            "highres_image",
            "network",
            {
                "identifiers": "\r".join(string_ids),
                "species": SPECIES,
                "required_score": REQUIRED_SCORE,
                "network_type": "functional",
                "add_nodes": 0,
                "network_flavor": "confidence",
                "hide_disconnected_nodes": 0,
            },
            binary=True,
        )


def save_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def save_enrichment(data, out_all: Path, out_sig: Path):
    df = pd.DataFrame(data)
    if not df.empty:
        for col in ["fdr", "p_value"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        sort_cols = [c for c in ["fdr", "p_value"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, na_position="last")

    df.to_csv(out_all, index=False, encoding="utf-8-sig")

    if not df.empty and "fdr" in df.columns:
        sig = df[df["fdr"] < 0.05].copy()
    else:
        sig = df.iloc[0:0].copy()

    sig.to_csv(out_sig, index=False, encoding="utf-8-sig")
    return df, sig


def network_stats(
    mapping: pd.DataFrame,
    network_data: list[dict],
    panel_name: str,
):
    degree = Counter()
    scores = defaultdict(list)

    for r in network_data:
        a = r.get("stringId_A")
        b = r.get("stringId_B")
        score = float(r.get("score", 0) or 0)

        if a:
            degree[a] += 1
            scores[a].append(score)
        if b:
            degree[b] += 1
            scores[b].append(score)

    rows = []
    for _, r in mapping.iterrows():
        sid = r["stringId"]
        vals = scores.get(sid, [])
        rows.append({
            "panel": panel_name,
            "gene_name": r["queryItem"],
            "string_id": sid,
            "string_preferred_name": r.get("preferredName", ""),
            "degree": int(degree.get(sid, 0)),
            "network_supported": int(degree.get(sid, 0) > 0),
            "max_edge_score": max(vals) if vals else 0.0,
            "mean_edge_score": sum(vals)/len(vals) if vals else 0.0,
        })

    node_df = pd.DataFrame(rows)
    n = len(node_df)
    e = len(network_data)
    possible = n * (n - 1) / 2 if n > 1 else 0
    density = e / possible if possible else 0.0

    summary = {
        "panel": panel_name,
        "mapped_nodes": n,
        "edges_score_ge_400": e,
        "genes_with_at_least_one_edge": int(
            node_df["network_supported"].sum()
        ) if not node_df.empty else 0,
        "mean_degree": float(node_df["degree"].mean()) if not node_df.empty else 0,
        "median_degree": float(node_df["degree"].median()) if not node_df.empty else 0,
        "max_degree": int(node_df["degree"].max()) if not node_df.empty else 0,
        "network_density": float(density),
    }
    return node_df, summary


def ppi_row(panel_name: str, ppi_data):
    row = ppi_data[0] if isinstance(ppi_data, list) and ppi_data else (
        ppi_data if isinstance(ppi_data, dict) else {}
    )
    out = {"panel": panel_name}
    out.update(row)
    return out


def main():
    AGENTIC = Path(__file__).resolve().parents[1]
    outdir = AGENTIC / "results" / "string_consensus_comparison"
    outdir.mkdir(parents=True, exist_ok=True)

    evidence_path = AGENTIC / "data" / "gene_evidence_master.csv"
    pubmed_path = AGENTIC / "data" / "pubmed_593_summary.csv"

    evidence = pd.read_csv(evidence_path).fillna("")
    pubmed = pd.read_csv(pubmed_path).fillna("")

    if len(evidence) != 593:
        raise ValueError(
            f"Expected 593 rows in evidence master, found {len(evidence)}"
        )
    if len(pubmed) != 593:
        raise ValueError(
            f"Expected 593 rows in PubMed summary, found {len(pubmed)}"
        )

    # Numeric normalization.
    evidence["support_top1000"] = pd.to_numeric(
        evidence["support_top1000"], errors="raise"
    ).astype(int)

    pubmed["has_error"] = pd.to_numeric(
        pubmed["has_error"], errors="coerce"
    ).fillna(0).astype(int)
    pubmed["total_hit_count"] = pd.to_numeric(
        pubmed["total_hit_count"], errors="coerce"
    )

    merged = evidence.merge(
        pubmed[
            [
                "gene_name",
                "total_hit_count",
                "retrieved_record_count",
                "pubmed_positive",
                "has_error",
                "error_message",
            ]
        ],
        on="gene_name",
        how="left",
        validate="one_to_one",
    )

    if merged["has_error"].isna().any():
        missing = merged.loc[
            merged["has_error"].isna(), "gene_name"
        ].tolist()
        raise ValueError(
            f"Genes missing from PubMed summary: {missing[:20]}"
        )

    def pubmed_status(row):
        if int(row["has_error"]) == 1:
            return "RETRIEVAL_ERROR"
        if pd.notna(row["total_hit_count"]) and row["total_hit_count"] > 0:
            return "PUBMED_POSITIVE"
        return "NO_DIRECT_QUERY_HIT"

    merged["pubmed_status"] = merged.apply(pubmed_status, axis=1)
    merged["pubmed_hit_count_clean"] = merged["total_hit_count"].fillna(0).astype(int)

    panels = {
        "41_5of5": merged[merged["support_top1000"] == 5].copy(),
        "185_ge4of5": merged[merged["support_top1000"] >= 4].copy(),
        "593_ge3of5": merged[merged["support_top1000"] >= 3].copy(),
    }

    expected = {"41_5of5": 41, "185_ge4of5": 185, "593_ge3of5": 593}
    for name, data in panels.items():
        if len(data) != expected[name]:
            raise ValueError(
                f"{name}: expected {expected[name]} genes, got {len(data)}"
            )

    # Save PubMed tier summary.
    pubmed_rows = []
    for name, data in panels.items():
        ok = data["pubmed_status"] != "RETRIEVAL_ERROR"
        pos = data["pubmed_status"] == "PUBMED_POSITIVE"
        zero = data["pubmed_status"] == "NO_DIRECT_QUERY_HIT"
        pubmed_rows.append({
            "panel": name,
            "n_genes": len(data),
            "successful_queries": int(ok.sum()),
            "retrieval_errors": int((~ok).sum()),
            "pubmed_positive": int(pos.sum()),
            "no_direct_query_hit": int(zero.sum()),
            "pubmed_positive_pct_successful": float(
                pos.sum()/ok.sum()*100 if ok.sum() else 0
            ),
            "sum_hit_count": int(
                data.loc[ok, "pubmed_hit_count_clean"].sum()
            ),
        })

    pd.DataFrame(pubmed_rows).to_csv(
        outdir / "pubmed_tier_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("STRING CONSENSUS-TIER COMPARISON")
    print("=" * 80)
    for row in pubmed_rows:
        print(
            f"{row['panel']}: n={row['n_genes']}, "
            f"PubMed+={row['pubmed_positive']}, "
            f"zero={row['no_direct_query_hit']}, "
            f"errors={row['retrieval_errors']}"
        )

    client = StringClient()

    # Map all 593 once; every smaller panel is a subset.
    print("\n[1] Mapping all 593 genes to STRING...")
    mapping593 = client.map_identifiers(
        panels["593_ge3of5"]["gene_name"].tolist()
    )
    mapping593.to_csv(
        outdir / "string_mapping_593.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mapped_gene_set = set(mapping593["queryItem"]) if not mapping593.empty else set()
    unmapped = merged[
        ~merged["gene_name"].isin(mapped_gene_set)
    ][["gene_name","support_top1000","tier","pubmed_status"]].copy()
    unmapped.to_csv(
        outdir / "string_unmapped_593.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gene_to_sid = dict(
        zip(mapping593["queryItem"], mapping593["stringId"])
    )

    # Build per-panel mappings from the single 593 mapping.
    panel_mappings = {}
    for name, data in panels.items():
        genes = set(data["gene_name"])
        panel_mappings[name] = mapping593[
            mapping593["queryItem"].isin(genes)
        ].copy()

    mapped_background_593 = list(
        dict.fromkeys(panel_mappings["593_ge3of5"]["stringId"].tolist())
    )

    print(
        f"Mapped 593 background proteins: {len(mapped_background_593)}; "
        f"unmapped genes: {len(unmapped)}"
    )

    network_summaries = []
    ppi_rows = []
    node_tables = {}

    # Identical network and PPI settings across all panels.
    for idx, (name, mapping) in enumerate(panel_mappings.items(), start=2):
        print(f"\n[{idx}] STRING network + PPI enrichment: {name}")
        ids = list(dict.fromkeys(mapping["stringId"].tolist()))
        if not ids:
            raise RuntimeError(f"No mapped STRING IDs for {name}")

        net = client.network(ids)
        pd.DataFrame(net).to_csv(
            outdir / f"string_network_edges_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        nodes, net_summary = network_stats(mapping, net, name)
        nodes.to_csv(
            outdir / f"string_nodes_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        node_tables[name] = nodes
        network_summaries.append(net_summary)

        ppi = client.ppi_enrichment(ids)
        save_json(
            outdir / f"string_ppi_enrichment_{name}.json",
            ppi,
        )
        ppi_rows.append(ppi_row(name, ppi))

        # Network images are useful for 41/185. The 593 image may be dense,
        # but attempt it and continue if STRING declines.
        try:
            image = client.network_image(ids)
            (outdir / f"string_network_{name}.png").write_bytes(image)
        except Exception as exc:
            save_json(
                outdir / f"string_network_image_error_{name}.json",
                {"error": type(exc).__name__, "message": str(exc)}
            )

    network_summary_df = pd.DataFrame(network_summaries)
    network_summary_df.to_csv(
        outdir / "string_network_summary_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ppi_df = pd.DataFrame(ppi_rows)
    ppi_df.to_csv(
        outdir / "string_ppi_enrichment_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Default-human-background enrichment for all three tiers
    # --------------------------------------------------------------
    sig_counts = []
    proteome_sig_tables = {}

    for name, mapping in panel_mappings.items():
        print(f"\nFunctional enrichment vs human background: {name}")
        ids = list(dict.fromkeys(mapping["stringId"].tolist()))
        data = client.enrichment(ids)
        all_df, sig_df = save_enrichment(
            data,
            outdir / f"string_enrichment_{name}_human_background.csv",
            outdir / f"string_enrichment_{name}_human_background_fdr05.csv",
        )
        proteome_sig_tables[name] = sig_df
        sig_counts.append({
            "panel": name,
            "significant_terms_fdr05_human_background": int(len(sig_df))
        })

    # --------------------------------------------------------------
    # Stricter tiers against the mapped 593 candidate background
    # --------------------------------------------------------------
    candidate_bg_counts = []

    for name in ["41_5of5", "185_ge4of5"]:
        print(f"\nCandidate-background enrichment: {name} vs mapped 593")
        ids = list(
            dict.fromkeys(panel_mappings[name]["stringId"].tolist())
        )
        try:
            data = client.enrichment(
                ids,
                background_ids=mapped_background_593,
            )
            _, sig_df = save_enrichment(
                data,
                outdir / f"string_enrichment_{name}_vs_593_background.csv",
                outdir / f"string_enrichment_{name}_vs_593_background_fdr05.csv",
            )
            candidate_bg_counts.append({
                "panel": name,
                "significant_terms_fdr05_vs_593_background": int(len(sig_df)),
                "status": "OK",
            })
        except Exception as exc:
            candidate_bg_counts.append({
                "panel": name,
                "significant_terms_fdr05_vs_593_background": None,
                "status": f"ERROR: {type(exc).__name__}: {exc}",
            })
            save_json(
                outdir / f"string_enrichment_{name}_vs_593_ERROR.json",
                {"error": type(exc).__name__, "message": str(exc)}
            )

    # --------------------------------------------------------------
    # Integrated gene-level table over all 593
    # Use 593-network degree for a single comparable network feature.
    # --------------------------------------------------------------
    nodes593 = node_tables["593_ge3of5"][
        [
            "gene_name","string_id","string_preferred_name",
            "degree","network_supported",
            "max_edge_score","mean_edge_score"
        ]
    ].rename(columns={
        "degree": "string_degree_in_593_network",
        "network_supported": "string_network_supported_in_593",
        "max_edge_score": "string_max_edge_score_in_593",
        "mean_edge_score": "string_mean_edge_score_in_593",
    })

    combined = merged.merge(
        nodes593,
        on="gene_name",
        how="left",
    )
    combined["string_mapped"] = combined["string_id"].notna().astype(int)

    for c in [
        "string_degree_in_593_network",
        "string_network_supported_in_593",
        "string_max_edge_score_in_593",
        "string_mean_edge_score_in_593",
    ]:
        combined[c] = pd.to_numeric(
            combined[c], errors="coerce"
        ).fillna(0)

    combined["in_panel_41"] = (combined["support_top1000"] == 5).astype(int)
    combined["in_panel_185"] = (combined["support_top1000"] >= 4).astype(int)
    combined["in_panel_593"] = 1

    # Simple evidence flags only; no arbitrary composite biological score.
    combined["pubmed_evidence_flag"] = (
        combined["pubmed_status"] == "PUBMED_POSITIVE"
    ).astype(int)
    combined["string_network_flag"] = (
        combined["string_network_supported_in_593"] > 0
    ).astype(int)

    def fast_group(row):
        if row["pubmed_status"] == "RETRIEVAL_ERROR":
            return (
                "STRING_SUPPORTED_PUBMED_ERROR"
                if row["string_network_flag"] == 1
                else "PUBMED_ERROR_NO_STRING_EDGE"
            )
        pub = row["pubmed_evidence_flag"] == 1
        net = row["string_network_flag"] == 1
        if pub and net:
            return "PUBMED_AND_STRING"
        if (not pub) and net:
            return "STRING_NETWORK_RESCUED"
        if pub and (not net):
            return "PUBMED_ONLY"
        return "NO_PUBMED_OR_CORE_NETWORK_EDGE"

    combined["fast_track_evidence_group"] = combined.apply(
        fast_group, axis=1
    )

    combined.to_csv(
        outdir / "combined_pubmed_string_593_gene_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Top PubMed genes (descriptive only).
    combined.sort_values(
        ["pubmed_hit_count_clean","support_top1000"],
        ascending=[False,False]
    ).head(50).to_csv(
        outdir / "top50_pubmed_hit_genes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Final summary.
    summary = {
        "settings": {
            "species": "Homo sapiens",
            "ncbi_taxon_id": SPECIES,
            "network_type": "functional",
            "required_score": REQUIRED_SCORE,
            "added_neighbors": 0,
            "fdr_threshold": 0.05,
        },
        "pubmed_tiers": pubmed_rows,
        "string_mapping": {
            "mapped_593": int(len(mapping593)),
            "unmapped_593": int(len(unmapped)),
        },
        "network_summary": network_summaries,
        "ppi_enrichment": ppi_rows,
        "proteome_enrichment_counts": sig_counts,
        "candidate_background_enrichment_counts": candidate_bg_counts,
        "integrated_group_counts": (
            combined["fast_track_evidence_group"]
            .value_counts()
            .to_dict()
        ),
        "important_note": (
            "PubMed hit counts are coarse automated retrieval signals. "
            "They are not verified evidence strengths. STRING associations "
            "are functional-association evidence and do not establish causality."
        ),
    }
    save_json(outdir / "analysis_summary.json", summary)

    # Markdown summary.
    lines = [
        "# STRING Comparison: 41 / 185 / 593 Consensus Panels",
        "",
        "## PubMed screening",
        "",
    ]
    for r in pubmed_rows:
        lines.append(
            f"- **{r['panel']}**: {r['pubmed_positive']} PubMed-positive, "
            f"{r['no_direct_query_hit']} zero-hit, "
            f"{r['retrieval_errors']} retrieval errors "
            f"({r['pubmed_positive_pct_successful']:.1f}% positive among successful queries)."
        )

    lines += [
        "",
        "## STRING network comparison",
        "",
    ]
    for r in network_summaries:
        lines.append(
            f"- **{r['panel']}**: {r['mapped_nodes']} mapped nodes, "
            f"{r['edges_score_ge_400']} edges, "
            f"{r['genes_with_at_least_one_edge']} connected genes, "
            f"mean degree {r['mean_degree']:.2f}, "
            f"density {r['network_density']:.5f}."
        )

    lines += [
        "",
        "## Interpretation boundary",
        "",
        "PubMed hit count is an automated literature-retrieval flag, not a verified evidence score. "
        "STRING edges represent functional associations and do not establish gene causality or diagnostic validity.",
        "",
        "## Next fast-track step",
        "",
        "Use `combined_pubmed_string_593_gene_evidence.csv` together with the enrichment and "
        "PPI summaries to freeze one compact evidence-supported candidate panel, then evaluate "
        "that panel with the existing trainval-only ML evaluator.",
    ]

    (outdir / "STRING_CONSENSUS_COMPARISON_SUMMARY.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Results: {outdir}")
    print("Next key file:")
    print(outdir / "combined_pubmed_string_593_gene_evidence.csv")


if __name__ == "__main__":
    main()
