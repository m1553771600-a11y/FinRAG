"""finrag.errors —— 库的异常层级。"""

from __future__ import annotations


class RAGError(Exception):
    """所有 finrag 异常的基类。"""

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class MilvusConnectionError(RAGError):
    """Milvus 向量库连接失败（可重试）。"""


class EmptyRetrieval(RAGError):
    """检索无结果。通常 retrieve 返回空列表而非抛此异常；仅供显式调用方使用。"""


class FileParseError(RAGError):
    """文档解析失败。"""


class InvalidInputError(RAGError):
    """输入不合法（不可重试）。"""


class LLMAPIError(RAGError):
    """LLM 调用失败（可重试）。"""
