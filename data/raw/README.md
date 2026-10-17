# Development corpus

Bundled PDFs are grouped by department. `manifest.csv`, `manifest.json` and `dataset_summary.json` describe the fixture corpus; `evaluation_queries.json` and `conflict_scenarios.json` provide development evaluation inputs.

Ingestion recursively discovers PDF files, including uppercase `.PDF` extensions. It derives metadata from filenames and parent folders; manifest roles, version status and other annotations are not enforced by retrieval.

Runtime uploads are saved into these department folders. They do not update the fixture manifests or labelled questions, and they can change subsequent retrieval benchmark results. Use a separate corpus directory with the evaluation runner when comparing a fixed dataset.

See the root [README](../../README.md) for setup, indexing and evaluation commands.
