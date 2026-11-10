from finrag import RAGConfig, Retriever
from finrag.types import RankedDocument


class FakeEmbedder:
    def encode_query(self, text):
        return [0.1] * 4, {1: 0.5}

    def encode(self, texts, batch_size=12):
        return [[0.1] * 4] * len(texts), [{1: 0.5}] * len(texts)


class FakeStore:
    def __init__(self):
        self.filters = None

    def build_filter(self, tenant_id, course_id=None):
        self.filters = (tenant_id, course_id)
        return f'tenant_id == "{tenant_id}"'

    def search(self, dense, sparse, top_k, filters=None):
        return [
            {"content": "A", "score": 0.6, "metadata": {"source_name": "s1"}},
            {"content": "B", "score": 0.4, "metadata": {"source_name": "s2"}},
        ]

    def upsert(self, chunks):
        return len(chunks)

    def delete(self, document_id):
        pass


class EmptyStore(FakeStore):
    def search(self, *a, **k):
        return []


class FakeReranker:
    def rerank(self, query, documents, top_k=3):
        ranked = [
            RankedDocument(d["content"], d["score"], i, d["metadata"])
            for i, d in enumerate(documents)
        ]
        ranked = sorted(ranked, key=lambda x: x.score, reverse=True)[:top_k]
        return ranked, (ranked[0].score if ranked else 0.0)


def test_retrieve_with_fakes():
    store = FakeStore()
    r = Retriever(
        RAGConfig(milvus_uri="x"),
        embedder=FakeEmbedder(), store=store, reranker=FakeReranker(),
    )
    docs, conf = r.retrieve("q", "tenant_default")
    assert len(docs) == 2
    assert conf == 0.6
    assert docs[0].content == "A"
    assert store.filters == ("tenant_default", None)


def test_retrieve_course_filter():
    store = FakeStore()
    r = Retriever(RAGConfig(milvus_uri="x"), embedder=FakeEmbedder(), store=store, reranker=FakeReranker())
    r.retrieve("q", "t", course_id="c1")
    assert store.filters == ("t", "c1")


def test_empty_retrieval_returns_empty():
    r = Retriever(RAGConfig(milvus_uri="x"), embedder=FakeEmbedder(), store=EmptyStore(), reranker=FakeReranker())
    docs, conf = r.retrieve("q", "t")
    assert docs == []
    assert conf == 0.0


def test_missing_embedder_path_raises():
    r = Retriever(RAGConfig(milvus_uri="x"))  # 未注入 embedder，也未给路径
    try:
        r.retrieve("q", "t")
        assert False, "应当抛出 InvalidInputError"
    except Exception as e:
        assert type(e).__name__ == "InvalidInputError"
