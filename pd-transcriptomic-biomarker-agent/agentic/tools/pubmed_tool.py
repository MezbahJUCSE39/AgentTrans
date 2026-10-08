"""
PubMed evidence retrieval tool for PD gene-prioritization experiments.

This module does NOT use an LLM. It queries NCBI E-utilities and caches
structured PubMed records locally for later use by an agent.

Environment variables
---------------------
Required:
    NCBI_EMAIL="your_email@example.com"

Optional:
    NCBI_API_KEY="..."
    NCBI_TOOL="mezbah_pd_agentic"

The client stays slightly below NCBI's documented request-rate ceilings.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import requests


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

PD_QUERY = (
    "(\"Parkinson Disease\"[MeSH Terms] "
    "OR \"Parkinson disease\"[Title/Abstract] "
    "OR \"Parkinson's disease\"[Title/Abstract])"
)


def _collapse_ws(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\\s+", " ", text).strip()


def _xml_text(element) -> str:
    if element is None:
        return ""
    return _collapse_ws("".join(element.itertext()))


def build_gene_pd_query(gene_name: str) -> str:
    """Conservative gene-symbol + PD query using Title/Abstract matching."""
    gene = gene_name.strip().replace('"', "")
    if not gene:
        raise ValueError("gene_name is empty")
    return f'("{gene}"[Title/Abstract]) AND {PD_QUERY}'


class PubMedClient:
    def __init__(
        self,
        cache_path: str | Path,
        *,
        email: str | None = None,
        api_key: str | None = None,
        tool: str | None = None,
        timeout: int = 30,
    ):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.email = email or os.getenv("NCBI_EMAIL")
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.tool = tool or os.getenv("NCBI_TOOL", "mezbah_pd_agentic")
        self.timeout = timeout

        if not self.email:
            raise ValueError(
                "NCBI contact email is required. Set NCBI_EMAIL before using this tool."
            )

        # NCBI documents 3 req/s without API key and 10 req/s with a key.
        self.min_interval = 0.12 if self.api_key else 0.36
        self._last_request_time = 0.0

        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        else:
            self.cache = {}

    def _save_cache(self):
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.cache_path)

    def _throttle(self):
        elapsed = time.monotonic() - self._last_request_time
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def _request(self, endpoint: str, params: dict[str, Any], method: str = "GET"):
        params = dict(params)
        params["tool"] = self.tool
        params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        self._throttle()
        url = f"{EUTILS}/{endpoint}"

        if method.upper() == "POST":
            response = requests.post(url, data=params, timeout=self.timeout)
        else:
            response = requests.get(url, params=params, timeout=self.timeout)

        self._last_request_time = time.monotonic()
        response.raise_for_status()
        return response

    @staticmethod
    def _cache_key(gene_name: str, max_records: int, sort: str) -> str:
        raw = json.dumps(
            {
                "gene_name": gene_name.upper(),
                "max_records": int(max_records),
                "sort": sort,
                "query_version": 1,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def search_gene_pd(
        self,
        gene_name: str,
        *,
        max_records: int = 10,
        sort: str = "relevance",
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Search PubMed for direct gene-symbol + Parkinson's disease evidence."""
        gene_name = gene_name.strip()
        if not gene_name:
            raise ValueError("gene_name is empty")
        if max_records < 1 or max_records > 100:
            raise ValueError("max_records must be between 1 and 100")

        key = self._cache_key(gene_name, max_records, sort)
        if not force_refresh and key in self.cache:
            result = dict(self.cache[key])
            result["cache_hit"] = True
            return result

        query = build_gene_pd_query(gene_name)

        search_resp = self._request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_records,
                "sort": sort,
            },
        )
        payload = search_resp.json()["esearchresult"]
        total_count = int(payload["count"])
        pmids = payload.get("idlist", [])

        records = []
        if pmids:
            fetch_resp = self._request(
                "efetch.fcgi",
                {
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "xml",
                },
                method="POST" if len(pmids) > 20 else "GET",
            )
            root = ET.fromstring(fetch_resp.content)

            for article in root.findall(".//PubmedArticle"):
                medline = article.find("MedlineCitation")
                pmid = _xml_text(medline.find("PMID")) if medline is not None else ""
                art = medline.find("Article") if medline is not None else None
                title = _xml_text(art.find("ArticleTitle")) if art is not None else ""

                abstract_parts = []
                if art is not None:
                    for node in art.findall("./Abstract/AbstractText"):
                        label = node.attrib.get("Label", "").strip()
                        body = _xml_text(node)
                        if label and body:
                            abstract_parts.append(f"{label}: {body}")
                        elif body:
                            abstract_parts.append(body)
                abstract = _collapse_ws(" ".join(abstract_parts))

                journal = ""
                pub_date = ""
                if art is not None:
                    journal = _xml_text(art.find("./Journal/Title"))
                    date_node = art.find("./Journal/JournalIssue/PubDate")
                    if date_node is not None:
                        year = _xml_text(date_node.find("Year"))
                        medline_date = _xml_text(date_node.find("MedlineDate"))
                        month = _xml_text(date_node.find("Month"))
                        day = _xml_text(date_node.find("Day"))
                        pub_date = " ".join(x for x in (year or medline_date, month, day) if x)

                authors = []
                if art is not None:
                    for author in art.findall("./AuthorList/Author"):
                        collective = _xml_text(author.find("CollectiveName"))
                        if collective:
                            authors.append(collective)
                            continue
                        last = _xml_text(author.find("LastName"))
                        initials = _xml_text(author.find("Initials"))
                        full = " ".join(x for x in (last, initials) if x)
                        if full:
                            authors.append(full)

                mesh_terms = []
                if medline is not None:
                    for heading in medline.findall("./MeshHeadingList/MeshHeading"):
                        term = _xml_text(heading.find("DescriptorName"))
                        if term:
                            mesh_terms.append(term)

                records.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "journal": journal,
                        "publication_date": pub_date,
                        "authors": authors,
                        "mesh_terms": mesh_terms,
                    }
                )

        result = {
            "gene_name": gene_name,
            "query": query,
            "total_hit_count": total_count,
            "retrieved_record_count": len(records),
            "pmids": pmids,
            "records": records,
            "retrieval": {
                "database": "PubMed",
                "source": "NCBI E-utilities",
                "sort": sort,
                "max_records": max_records,
                "tool": self.tool,
            },
            "cache_hit": False,
        }

        self.cache[key] = result
        self._save_cache()
        return result


def search_pubmed_pd(
    gene_name: str,
    *,
    cache_path: str | Path = "agentic/data/pubmed_cache.json",
    max_records: int = 10,
) -> dict[str, Any]:
    """Convenience function for later wrapping as an agent tool."""
    return PubMedClient(cache_path=cache_path).search_gene_pd(
        gene_name,
        max_records=max_records,
    )


def batch_search_pubmed_pd(
    gene_names: Iterable[str],
    *,
    cache_path: str | Path = "agentic/data/pubmed_cache.json",
    max_records: int = 10,
) -> list[dict[str, Any]]:
    """Sequential, resumable batch retrieval with per-gene error capture."""
    client = PubMedClient(cache_path=cache_path)
    results = []

    for i, gene in enumerate(gene_names, start=1):
        gene = str(gene).strip()
        if not gene:
            continue

        print(f"[{i}] PubMed: {gene}", flush=True)
        try:
            result = client.search_gene_pd(gene, max_records=max_records)
        except Exception as exc:
            result = {
                "gene_name": gene,
                "error": type(exc).__name__,
                "message": str(exc),
            }
        results.append(result)

    return results
