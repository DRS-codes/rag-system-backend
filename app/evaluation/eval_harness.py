"""
End-to-end evaluation harness.

Runs every question in an eval set through the live pipeline (retrieval
+ generation), computes retrieval metrics against ground-truth doc_ids
and generation metrics via LLM-as-judge, and aggregates into a report.

This is what turns "we built a RAG system" into "we can show retrieval
precision@4 is 0.82 and faithfulness is 0.91, and here's the run that
regressed when we switched chunk size from 512 to 256" — i.e. an actual
engineering feedback loop instead of eyeballing a few demo queries.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from app.config import settings
from app.evaluation.metrics import answer_relevancy, faithfulness, retrieval_metrics
from app.generation.generator import generate_answer
from app.models.schemas import EvalExample, EvalReport
from app.retrieval.pipeline import RagPipeline


def load_eval_set(path: str) -> list[EvalExample]:
    with open(path) as f:
        raw = json.load(f)
    return [EvalExample(**row) for row in raw]


def run_eval(pipeline: RagPipeline, eval_set: list[EvalExample], k: int | None = None, judge_generation: bool = True) -> EvalReport:
    k = k or settings.final_top_k
    per_example = []

    for ex in eval_set:
        retrieved = pipeline.retrieve(ex.question, final_top_k=k)
        r_metrics = retrieval_metrics(retrieved, set(ex.ground_truth_doc_ids), k)

        row = {
            "question": ex.question,
            "ground_truth_doc_ids": ex.ground_truth_doc_ids,
            "retrieved_doc_ids": [sc.chunk.doc_id for sc in retrieved],
            "retrieval": r_metrics,
        }

        if judge_generation:
            answer = generate_answer(ex.question, retrieved)
            f_result = faithfulness(answer, retrieved)
            rel_result = answer_relevancy(ex.question, answer)
            row.update({
                "answer": answer,
                "reference_answer": ex.reference_answer,
                "faithfulness": f_result,
                "answer_relevancy": rel_result,
            })

        per_example.append(row)

    retrieval_agg = {
        metric: round(mean(r["retrieval"][metric] for r in per_example), 4)
        for metric in ["precision@k", "recall@k", "mrr", "ndcg@k"]
    }

    generation_agg = {}
    if judge_generation:
        faith_scores = [r["faithfulness"]["score"] for r in per_example if r["faithfulness"].get("score") is not None]
        rel_scores = [r["answer_relevancy"]["score"] for r in per_example if r["answer_relevancy"].get("score") is not None]
        generation_agg = {
            "faithfulness": round(mean(faith_scores), 4) if faith_scores else None,
            "answer_relevancy": round(mean(rel_scores), 4) if rel_scores else None,
            "judged_examples": len(faith_scores),
        }

    return EvalReport(
        num_examples=len(eval_set),
        retrieval=retrieval_agg,
        generation=generation_agg,
        per_example=per_example,
    )


def write_report(report: EvalReport, out_dir: str) -> tuple[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "eval_report.json"
    with open(json_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    md_path = out / "eval_report.md"
    lines = [
        "# RAG Evaluation Report",
        "",
        f"**Examples evaluated:** {report.num_examples}",
        "",
        "## Retrieval",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k_, v in report.retrieval.items():
        lines.append(f"| {k_} | {v} |")

    if report.generation:
        lines += ["", "## Generation (LLM-as-judge)", "", "| Metric | Value |", "|---|---|"]
        for k_, v in report.generation.items():
            lines.append(f"| {k_} | {v} |")

    lines += ["", "## Per-example detail", ""]
    for row in report.per_example:
        lines.append(f"### {row['question']}")
        lines.append(f"- Retrieval: {row['retrieval']}")
        if "faithfulness" in row:
            lines.append(f"- Faithfulness: {row['faithfulness'].get('score')}")
            lines.append(f"- Answer relevancy: {row['answer_relevancy'].get('score')}")
            lines.append(f"- Answer: {row['answer'][:300]}")
        lines.append("")

    md_path.write_text("\n".join(lines))
    return str(json_path), str(md_path)
