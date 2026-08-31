from app.ingestion.chunking import FixedSizeChunker, RecursiveChunker, count_tokens

SAMPLE = """Paragraph one talks about apples. Apples are a fruit. They grow on trees.

Paragraph two talks about cars. Cars have engines. Engines burn fuel.

Paragraph three talks about oceans. Oceans are deep. They cover most of Earth."""


def test_fixed_chunker_respects_size():
    chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk_document("doc1", SAMPLE)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= 20


def test_fixed_chunker_overlap_present():
    chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.split(SAMPLE)
    assert len(chunks) >= 2
    # last few words of chunk 0 should reappear at the start of chunk 1
    tail = chunks[0].split()[-3:]
    assert any(word in chunks[1] for word in tail)


def test_recursive_chunker_keeps_paragraphs_together_when_small():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.split(SAMPLE)
    # small doc, generous chunk_size -> should collapse to very few chunks
    assert len(chunks) <= 2


def test_recursive_chunker_splits_large_doc():
    long_text = "\n\n".join([f"Section {i}. " + ("word " * 200) for i in range(5)])
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split(long_text)
    assert len(chunks) > 5
    for c in chunks:
        assert count_tokens(c) <= 130  # small slack for packing edge cases


def test_chunk_ids_are_unique():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)
    chunks = chunker.chunk_document("doc1", SAMPLE)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
