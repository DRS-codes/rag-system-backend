from app.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from app.models.schemas import Chunk, ScoredChunk


def _scored(doc_id: str, score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(chunk_id=f"{doc_id}_c", doc_id=doc_id, text="x", position=0, token_count=1)
    return ScoredChunk(chunk=chunk, score=score, source="hybrid")


def test_precision_at_k_all_relevant():
    retrieved = [_scored("a"), _scored("b")]
    assert precision_at_k(retrieved, {"a", "b"}, k=2) == 1.0


def test_precision_at_k_partial():
    retrieved = [_scored("a"), _scored("z")]
    assert precision_at_k(retrieved, {"a"}, k=2) == 0.5


def test_recall_at_k_finds_all_relevant_docs():
    retrieved = [_scored("a"), _scored("b"), _scored("c")]
    assert recall_at_k(retrieved, {"a", "b"}, k=3) == 1.0


def test_recall_at_k_misses_relevant_doc_outside_k():
    retrieved = [_scored("z"), _scored("a")]
    assert recall_at_k(retrieved, {"a", "b"}, k=1) == 0.0  # b never retrieved, a outside k=1


def test_mrr_rewards_early_hit():
    retrieved_early = [_scored("a"), _scored("z")]
    retrieved_late = [_scored("z"), _scored("a")]
    assert mrr(retrieved_early, {"a"}) == 1.0
    assert mrr(retrieved_late, {"a"}) == 0.5


def test_mrr_zero_when_never_found():
    retrieved = [_scored("z"), _scored("y")]
    assert mrr(retrieved, {"a"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    retrieved = [_scored("a"), _scored("b")]
    assert ndcg_at_k(retrieved, {"a", "b"}, k=2) == 1.0


def test_ndcg_penalizes_worse_ranking():
    ideal = [_scored("a"), _scored("z")]
    worse = [_scored("z"), _scored("a")]
    assert ndcg_at_k(ideal, {"a"}, k=2) > ndcg_at_k(worse, {"a"}, k=2)
