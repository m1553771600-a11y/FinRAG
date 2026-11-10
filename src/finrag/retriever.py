"""finrag.retriever —— 检索 pipeline（encode -> hybrid -> rerank）。

等价契约: ``Retriever.retrieve`` 的参数名/顺序/默认值与旧
``backend.core.reranker.retrieve()`` 一致。
"""

from __future__ import annotations

from typing import Optional

from .config import RAGConfig
from .types import RankedDocument, Embedder, Reranker, VectorStore
from .errors import InvalidInputError
from ._logging import get_logger

logger = get_logger(__name__)


class Retriever:
    """Hybrid 召回 -> 精排 一体化检索。

    用法::

        retriever = Retriever(config)
        docs, confidence = retriever.retrieve(query, tenant_id, course_id)
    """

    def __init__(
        self,
        config: RAGConfig,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        store: VectorStore | None = None,
    ):
        self._config = config
        self._embedder = embedder
        self._reranker = reranker
        self._store = store

    # ── 惰性构造默认实现（重依赖只在真正使用时才 import）─────────

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            from .embedder import BGE3Embedder

            if not self._config.embedding_model_path:
                raise InvalidInputError("未提供 embedding_model_path，也未注入 embedder")
            self._embedder = BGE3Embedder(self._config.embedding_model_path)
        return self._embedder

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            from .reranker import BGERerankerImpl

            self._reranker = BGERerankerImpl(
                self._config.reranker_model_path,
                max_input_chars=self._config.rerank_max_input_chars,
            )
        return self._reranker

    def _get_store(self) -> VectorStore:
        if self._store is None:
            from .vector_store import MilvusStore

            self._store = MilvusStore(
                self._config.milvus_uri,
                self._config.collection_name,
                ann_ef=self._config.ann_ef,
                dense_weight=self._config.dense_weight,
                sparse_weight=self._config.sparse_weight,
            )
        return self._store

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        course_id: str | None = None,
        *,
        recall_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> tuple[list[RankedDocument], float]:
        """Hybrid 召回 -> 精排一体化 Pipeline。

        Returns:
            (ranked_docs, confidence)
            ranked_docs: 精排后 Top-rerank_top_k 文档
            confidence:  Top-1 文档的置信度 [0, 1]；空检索返回 ([] , 0.0)
        """
        recall_top_k = recall_top_k if recall_top_k is not None else self._config.recall_top_k
        rerank_top_k = rerank_top_k if rerank_top_k is not None else self._config.rerank_top_k

        embedder = self._get_embedder()
        dense, sparse = embedder.encode_query(query)

        store = self._get_store()
        filters = store.build_filter(tenant_id, course_id)
        candidates = store.search(dense, sparse, top_k=recall_top_k, filters=filters)

        if not candidates:
            logger.info("retrieve.empty", query_preview=query[:50])
            return [], 0.0

        reranker = self._get_reranker()
        return reranker.rerank(query, candidates, top_k=rerank_top_k)
