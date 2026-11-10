"""finrag —— 领域无关的金融 RAG 库。

从 ClipAgent_Pro 抽取而来，剥离了剪辑领域知识，RAG 核心领域无关，
并内置 ``finrag.domains.finance`` 金融领域 profile。

公共 API:
    RAGConfig          —— 显式配置对象（库内部不读全局/环境）
    Retriever          —— Hybrid 召回 + 精排 pipeline
    Ingestor           —— 文档加载/分块/嵌入/入库
    AnswerGenerator    —— 拼 prompt -> LLM -> 回答（prompt 可注入）
    QueryClassifier    —— 意图二分类（可选）
    DocumentChunk / RankedDocument —— 数据结构
"""

from .config import RAGConfig, load_config
from .types import (
    DocumentChunk,
    RankedDocument,
    Embedder,
    Reranker,
    VectorStore,
    ChatLLM,
    generate_chunk_id,
)
from .errors import (
    RAGError,
    MilvusConnectionError,
    EmptyRetrieval,
    FileParseError,
    InvalidInputError,
    LLMAPIError,
)
from .retriever import Retriever
from .ingestor import Ingestor
from .answer import AnswerGenerator, Answer
from .classifier import QueryClassifier
from .llm import OpenAICompatLLM
from .web import WebSearch, DuckDuckGoSearch, TavilySearch, search_web

__version__ = "0.1.0"

__all__ = [
    "RAGConfig",
    "load_config",
    "Retriever",
    "Ingestor",
    "AnswerGenerator",
    "Answer",
    "QueryClassifier",
    "OpenAICompatLLM",
    "WebSearch",
    "DuckDuckGoSearch",
    "TavilySearch",
    "search_web",
    "DocumentChunk",
    "RankedDocument",
    "Embedder",
    "Reranker",
    "VectorStore",
    "ChatLLM",
    "generate_chunk_id",
    "RAGError",
    "MilvusConnectionError",
    "EmptyRetrieval",
    "FileParseError",
    "InvalidInputError",
    "LLMAPIError",
    "__version__",
]
