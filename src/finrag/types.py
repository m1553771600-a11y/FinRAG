"""finrag.types —— 数据结构、依赖注入协议（Protocol）与公共工具函数。"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Protocol


# ──────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """待写入向量库的单个文档块，字段与 Milvus Schema 一一对应。

    id:               全局唯一 ID（MD5 of content + document_id + chunk_index）
    content:          chunk 文本
    embedding:        Dense 向量（BGE-M3，1024 维）
    sparse_embedding: Sparse 向量（{token_id: weight}）
    source_name:      来源标注（如 "研报 > 估值 > PE"）
    """

    id: str
    content: str
    embedding: list[float]
    sparse_embedding: dict
    course_id: str
    document_id: str
    source_name: str
    chunk_type: str
    chunk_index: int
    version: str
    tenant_id: str = "tenant_default"
    updated_at: int = field(default_factory=lambda: int(time.time()))


@dataclass
class RankedDocument:
    """精排后的单个文档结果。"""

    content: str
    score: float
    original_index: int
    metadata: dict


def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
    """生成 chunk 全局唯一 ID（MD5）。内容+位置不变则 ID 不变，支持幂等 upsert。"""
    raw = f"{document_id}_{chunk_index}_{content[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────
# 依赖注入协议（Protocol）—— 允许用户替换任意实现
# ──────────────────────────────────────────────────────────────

class Embedder(Protocol):
    """嵌入器协议。实现：BGE-M3 / OpenAI embedding / 任意。"""

    def encode_query(self, text: str) -> tuple[list[float], dict]: ...
    def encode(self, texts: list[str], batch_size: int = 12) -> tuple[list[list[float]], list[dict]]: ...


class Reranker(Protocol):
    """精排器协议。返回 (ranked_docs, confidence)。"""

    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> tuple[list[RankedDocument], float]: ...


class VectorStore(Protocol):
    """向量库协议。实现：Milvus / Qdrant / 任意。"""

    def upsert(self, chunks: list[DocumentChunk]) -> int: ...
    def delete(self, document_id: str) -> None: ...
    def search(self, dense: list[float], sparse: dict, top_k: int, filters: str | None) -> list[dict]: ...
    def build_filter(self, tenant_id: str, course_id: str | None = None) -> str: ...


class ChatLLM(Protocol):
    """对话 LLM 协议（langchain 兼容的最小面）。"""

    def invoke(self, messages: list) -> object: ...
    async def ainvoke(self, messages: list) -> object: ...
