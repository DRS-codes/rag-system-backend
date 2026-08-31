from app.models.schemas import Chunk
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.keyword_search import KeywordStore, tokenize


def _chunk(cid: str, text: str = "text") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=f"doc_{cid}", text=text, position=0, token_count=1)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_keyword_store_finds_exact_term():
    store = KeywordStore()
    store.add([
        _chunk("a", "the quick brown fox jumps over the lazy dog"),
        _chunk("b", "completely unrelated text about oceans and tides"),
    ])
    results = store.search("fox", top_k=5)
    assert results
    assert results[0][0].chunk_id == "a"


def test_rrf_boosts_items_ranked_highly_in_both_lists():
    a1, a2, a3 = _chunk("a1"), _chunk("a2"), _chunk("a3")

    vector_results = [(a1, 0.9), (a2, 0.8), (a3, 0.7)]
    keyword_results = [(a2, 10.0), (a1, 8.0), (a3, 1.0)]

    fused = reciprocal_rank_fusion([vector_results, keyword_results], k=60)
    fused_ids = [sc.chunk.chunk_id for sc in fused]

    # a1 and a2 each appear in the top 2 of both lists, a3 is always last -> a3 should rank lowest
    assert fused_ids[-1] == "a3"
    assert set(fused_ids[:2]) == {"a1", "a2"}


def test_rrf_handles_item_present_in_only_one_list():
    a1 = _chunk("a1")
    only_vector = _chunk("only_vector")
    only_keyword = _chunk("only_keyword")

    vector_results = [(a1, 0.9), (only_vector, 0.5)]
    keyword_results = [(a1, 5.0), (only_keyword, 2.0)]

    fused = reciprocal_rank_fusion([vector_results, keyword_results], k=60)
    fused_ids = {sc.chunk.chunk_id for sc in fused}
    assert fused_ids == {"a1", "only_vector", "only_keyword"}
    # a1 appears in both lists at rank 0 -> should score highest
    assert fused[0].chunk.chunk_id == "a1"
