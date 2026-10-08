# STRING Comparison: 41 / 185 / 593 Consensus Panels

## PubMed screening

- **41_5of5**: 16 PubMed-positive, 25 zero-hit, 0 retrieval errors (39.0% positive among successful queries).
- **185_ge4of5**: 83 PubMed-positive, 102 zero-hit, 0 retrieval errors (44.9% positive among successful queries).
- **593_ge3of5**: 263 PubMed-positive, 329 zero-hit, 1 retrieval errors (44.4% positive among successful queries).

## STRING network comparison

- **41_5of5**: 39 mapped nodes, 7 edges, 12 connected genes, mean degree 0.36, density 0.00945.
- **185_ge4of5**: 182 mapped nodes, 225 edges, 109 connected genes, mean degree 2.47, density 0.01366.
- **593_ge3of5**: 583 mapped nodes, 2399 edges, 466 connected genes, mean degree 8.23, density 0.01414.

## Interpretation boundary

PubMed hit count is an automated literature-retrieval flag, not a verified evidence score. STRING edges represent functional associations and do not establish gene causality or diagnostic validity.

## Next fast-track step

Use `combined_pubmed_string_593_gene_evidence.csv` together with the enrichment and PPI summaries to freeze one compact evidence-supported candidate panel, then evaluate that panel with the existing trainval-only ML evaluator.