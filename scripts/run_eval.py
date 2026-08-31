"""
Run the full ingest -> eval loop from the command line, no server needed.
Useful for quick iteration when tuning chunk_size / chunk_strategy /
rerank_enabled in .env — run this, check eval_reports/eval_report.md,
change a setting, run again.

Usage: python -m scripts.run_eval
"""
from __future__ import annotations

from app.config import settings
from app.evaluation.eval_harness import load_eval_set, run_eval, write_report
from app.ingestion.loader import load_directory
from app.retrieval.pipeline import RagPipeline


def main():
    pipeline = RagPipeline()

    if RagPipeline.exists(settings.index_dir):
        pipeline.load(settings.index_dir)
        print(f"Loaded existing index ({len(pipeline.vector_store.chunks)} chunks)")
    else:
        print("Building index from data/sample_docs/ ...")
        for doc_id, text, metadata in load_directory("data/sample_docs"):
            chunks = pipeline.ingest(doc_id, text, metadata)
            print(f"  {doc_id}: {len(chunks)} chunks ({settings.chunk_strategy} strategy)")
        pipeline.save(settings.index_dir)

    eval_set = load_eval_set("data/eval_set.json")
    print(f"\nRunning eval on {len(eval_set)} questions...")
    report = run_eval(pipeline, eval_set)

    json_path, md_path = write_report(report, out_dir="./eval_reports")
    print(f"\nRetrieval metrics: {report.retrieval}")
    print(f"Generation metrics: {report.generation}")
    print(f"\nFull report written to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
