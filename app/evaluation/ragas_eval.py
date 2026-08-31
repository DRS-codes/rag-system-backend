"""
Optional: run RAGAS on the same eval set for a second, independently-
implemented opinion on faithfulness/relevancy (and RAGAS's own
context_precision / context_recall metrics). Not run by default —
RAGAS pulls in its own LLM calls and a `datasets` dependency, so it's
opt-in via RAGAS_ENABLED so the core eval loop stays fast and dependency
-light. Useful as a sanity check that the custom LLM-as-judge metrics in
metrics.py aren't systematically biased (e.g. the same judge model
scoring its own generations leniently).
"""
from __future__ import annotations

from app.evaluation.eval_harness import load_eval_set
from app.retrieval.pipeline import RagPipeline
from app.generation.generator import generate_answer


def run_ragas_eval(pipeline: RagPipeline, eval_set_path: str, k: int = 4) -> dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    eval_set = load_eval_set(eval_set_path)
    questions, answers, contexts, ground_truths = [], [], [], []

    for ex in eval_set:
        retrieved = pipeline.retrieve(ex.question, final_top_k=k)
        answer = generate_answer(ex.question, retrieved)
        questions.append(ex.question)
        answers.append(answer)
        contexts.append([sc.chunk.text for sc in retrieved])
        ground_truths.append(ex.reference_answer)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    return result.to_pandas().to_dict(orient="records")
