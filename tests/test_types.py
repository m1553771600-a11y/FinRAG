from finrag.types import generate_chunk_id, DocumentChunk, RankedDocument


def test_generate_chunk_id_stable():
    a = generate_chunk_id("hello", "doc1", 0)
    assert a == generate_chunk_id("hello", "doc1", 0)
    assert a != generate_chunk_id("hello", "doc1", 1)
    assert a != generate_chunk_id("world", "doc1", 0)
    assert a != generate_chunk_id("hello", "doc2", 0)


def test_document_chunk_defaults():
    c = DocumentChunk(
        id="x", content="c", embedding=[0.0], sparse_embedding={},
        course_id="k", document_id="d", source_name="s", chunk_type="text",
        chunk_index=0, version="1.0",
    )
    assert c.tenant_id == "tenant_default"
    assert c.updated_at > 0


def test_ranked_document_fields():
    d = RankedDocument("c", 0.9, 1, {"a": 1})
    assert d.content == "c" and d.score == 0.9 and d.original_index == 1
