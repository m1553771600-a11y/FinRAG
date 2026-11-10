"""finrag.vector_store —— Milvus 向量库（默认实现）。

依赖: ``pymilvus``（base 依赖）。
包含 collection schema / 索引构建（对应旧 init_milvus.py）。
"""

from __future__ import annotations

from typing import Optional

from .types import DocumentChunk, generate_chunk_id
from .errors import MilvusConnectionError
from ._logging import get_logger

logger = get_logger(__name__)

VECTOR_DIM = 1024  # BGE-M3 稠密向量维度


class MilvusStore:
    """Milvus 知识库客户端。

    用法::

        store = MilvusStore("http://localhost:19530", "finance_knowledge")
        store.ensure_collection()
        store.upsert(chunks)
        candidates = store.search(dense, sparse, top_k=10, filters=store.build_filter(tenant_id))
    """

    def __init__(
        self,
        uri: str,
        collection_name: str,
        *,
        ann_ef: int = 64,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        from pymilvus import MilvusClient

        self._uri = uri
        self._collection_name = collection_name
        self._ann_ef = ann_ef
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        try:
            self._client = MilvusClient(uri=uri)
        except Exception as e:  # noqa: BLE001
            raise MilvusConnectionError(
                f"Milvus 连接失败: {uri}", details={"error": str(e)}
            ) from e
        logger.info("milvus.connected", uri=uri)

    # ── Schema / 索引（对应旧 init_milvus.py）────────────────────

    def ensure_collection(self, *, drop_existing: bool = False) -> None:
        """构建集合 schema + 索引（幂等）。drop_existing=True 会先删旧集合重建。"""
        from pymilvus import DataType

        client = self._client
        name = self._collection_name
        if client.has_collection(name):
            if drop_existing:
                client.drop_collection(name)
            else:
                return

        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("content", DataType.VARCHAR, max_length=4096)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("document_id", DataType.VARCHAR, max_length=64)
        schema.add_field("course_id", DataType.VARCHAR, max_length=64)
        schema.add_field("source_name", DataType.VARCHAR, max_length=256)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=32)
        schema.add_field("version", DataType.VARCHAR, max_length=32)
        schema.add_field("updated_at", DataType.INT64)

        ip = client.prepare_index_params()
        ip.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                     params={"M": 16, "efConstruction": 256})
        ip.add_index(field_name="sparse_embedding", index_type="SPARSE_INVERTED_INDEX",
                     metric_type="IP", params={"drop_ratio_build": 0.2})
        ip.add_index(field_name="tenant_id", index_type="INVERTED")
        ip.add_index(field_name="course_id", index_type="INVERTED")

        client.create_collection(collection_name=name, schema=schema, index_params=ip)
        logger.info("milvus.collection_created", collection=name)

    # ── 写入 ────────────────────────────────────────────────────

    def upsert(self, chunks: list[DocumentChunk]) -> int:
        """批量写入文档块（Upsert）。返回写入条数。"""
        if not chunks:
            return 0
        data = [
            {
                "id": c.id,
                "embedding": c.embedding,
                "sparse_embedding": c.sparse_embedding,
                "content": c.content[:4096],
                "chunk_index": c.chunk_index,
                "document_id": c.document_id,
                "course_id": c.course_id,
                "tenant_id": c.tenant_id,
                "source_name": c.source_name,
                "chunk_type": c.chunk_type,
                "version": c.version,
                "updated_at": c.updated_at,
            }
            for c in chunks
        ]
        self._client.upsert(collection_name=self._collection_name, data=data)
        self._flush()
        logger.info("knowledge_base.chunks_upserted", count=len(chunks))
        return len(chunks)

    def _flush(self) -> None:
        """强制 flush，确保数据立即可见 / 持久化（Milvus 默认异步落盘）。"""
        try:
            from pymilvus import Collection

            Collection(self._collection_name, using=self._client._using).flush()
        except Exception as e:  # noqa: BLE001
            logger.warning("milvus.flush_failed", error=str(e))

    def delete(self, document_id: str) -> None:
        """删除指定文档的所有 chunk（对 document_id 转义防注入）。"""
        safe_id = document_id.replace('"', '\\"')
        self._client.delete(
            collection_name=self._collection_name,
            filter=f'document_id == "{safe_id}"',
        )
        logger.info("knowledge_base.document_deleted", document_id=document_id)

    # ── 检索 ────────────────────────────────────────────────────

    def search(
        self,
        dense: list[float],
        sparse: dict,
        top_k: int,
        filters: str | None = None,
    ) -> list[dict]:
        """Dense + Sparse 双路 Hybrid 检索，WeightedRanker 融合。

        返回候选列表，每项含 "content" / "score" / "metadata"。
        """
        from pymilvus import AnnSearchRequest, WeightedRanker

        dense_req = AnnSearchRequest(
            data=[dense],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": self._ann_ef}},
            limit=top_k,
            expr=filters,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse],
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=top_k,
            expr=filters,
        )
        output_fields = [
            "content", "source_name", "chunk_type",
            "course_id", "document_id", "chunk_index",
        ]

        results = self._client.hybrid_search(
            collection_name=self._collection_name,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(self._dense_weight, self._sparse_weight),
            limit=top_k,
            output_fields=output_fields,
        )

        candidates = []
        for hit in results[0]:
            candidates.append({
                "content": hit["entity"].get("content") or "",
                "score": hit.get("distance") or 0.0,
                "metadata": {
                    "source_name": hit["entity"].get("source_name") or "",
                    "chunk_type": hit["entity"].get("chunk_type") or "text",
                    "course_id": hit["entity"].get("course_id") or "",
                    "document_id": hit["entity"].get("document_id") or "",
                    "chunk_index": hit["entity"].get("chunk_index") or 0,
                },
            })
        logger.info("knowledge_base.hybrid_search_done", candidates=len(candidates))
        return candidates

    @staticmethod
    def build_filter(tenant_id: str, course_id: Optional[str] = None) -> str:
        """构建 Milvus bool 过滤表达式（转义防注入）。"""
        safe_tenant = tenant_id.replace('"', '\\"')
        expr = f'tenant_id == "{safe_tenant}"'
        if course_id:
            safe_course = course_id.replace('"', '\\"')
            expr += f' and course_id == "{safe_course}"'
        return expr

    @staticmethod
    def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
        """生成 chunk 唯一 ID（MD5）。"""
        return generate_chunk_id(content, document_id, chunk_index)

    # ── 兼容旧命名（drop-in 等价）────────────────────────────────

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """[兼容别名] 见 :meth:`upsert`。"""
        return self.upsert(chunks)

    def delete_document_chunks(self, document_id: str) -> None:
        """[兼容别名] 见 :meth:`delete`。"""
        return self.delete(document_id)

    def _hybrid_search(
        self,
        query_embedding: list[float],
        query_sparse: dict,
        top_k: int,
        filters: Optional[str] = None,
    ) -> list[dict]:
        """[兼容别名] 见 :meth:`search`。"""
        return self.search(query_embedding, query_sparse, top_k, filters)

    def _build_filter(self, tenant_id: str, course_id: Optional[str] = None) -> str:
        """[兼容别名] 见 :meth:`build_filter`。"""
        return self.build_filter(tenant_id, course_id)
