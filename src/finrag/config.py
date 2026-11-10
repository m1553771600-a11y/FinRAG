"""finrag.config —— 显式配置对象。

设计约束（库形态的关键）:
    - 库内部**绝不**读取全局变量、环境变量或任何隐式配置；
    - 所有配置必须由调用方显式构造 :class:`RAGConfig` 传入。

为方便应用/示例，提供**显式**的加载辅助（库核心不会自动调用）：
    :meth:`RAGConfig.from_env_file` / :meth:`RAGConfig.from_env` / :func:`load_config`。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    """RAG 全部可配置项。不可变（frozen），按需通过 ``dataclasses.replace`` 派生。"""

    # ── 向量库（Milvus）────────────────────────────────────────
    milvus_uri: str = "http://localhost:19530"   # Milvus 地址
    collection_name: str = "finance_knowledge"   # 集合名

    # ── 本地模型权重路径（None 表示该组件需由调用方注入实现）────
    embedding_model_path: str | None = None      # BGE-M3 路径
    reranker_model_path: str | None = None       # BGE-Reranker 路径
    classifier_model_path: str | None = None     # 意图分类器路径（可选）

    # ── LLM（OpenAI 兼容：DeepSeek / OpenAI / 任意）─────────────
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "deepseek-chat"

    # ── 检索参数 ───────────────────────────────────────────────
    recall_top_k: int = 10
    rerank_top_k: int = 3
    ann_ef: int = 64
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rerank_max_input_chars: int = 512
    confidence_threshold: float = 0.75

    # ── 杂项 ───────────────────────────────────────────────────
    log_level: str = "INFO"

    @classmethod
    def from_mapping(cls, values: dict[str, str]) -> "RAGConfig":
        """从键值映射构造配置（FINRAG_* 优先，兼容旧 MILVUS_*/DEEPSEEK_* 键）。"""

        def _get(*keys: str) -> str | None:
            for k in keys:
                if values.get(k):
                    return values[k]
            return None

        milvus_uri = _get("FINRAG_MILVUS_URI")
        if not milvus_uri:
            host = _get("FINRAG_MILVUS_HOST", "MILVUS_HOST") or "localhost"
            port = _get("FINRAG_MILVUS_PORT", "MILVUS_PORT") or "19530"
            milvus_uri = f"http://{host}:{port}"

        return cls(
            milvus_uri=milvus_uri,
            collection_name=_get("FINRAG_COLLECTION_NAME", "RAG_COLLECTION_NAME") or "finance_knowledge",
            embedding_model_path=_get("FINRAG_EMBEDDING_MODEL_PATH", "BGE_M3_MODEL_PATH", "EMBEDDING_MODEL_PATH"),
            reranker_model_path=_get("FINRAG_RERANKER_MODEL_PATH", "RERANKER_MODEL_PATH"),
            classifier_model_path=_get("FINRAG_CLASSIFIER_MODEL_PATH", "FINETUNED_CLASSIFIER_PATH", "CLASSIFIER_MODEL_PATH"),
            llm_api_key=_get("FINRAG_LLM_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"),
            llm_base_url=_get("FINRAG_LLM_BASE_URL", "DEEPSEEK_BASE_URL", "LLM_BASE_URL"),
            llm_model=_get("FINRAG_LLM_MODEL", "DEEPSEEK_MODEL_CHAT", "LLM_MODEL") or "deepseek-chat",
            log_level=_get("FINRAG_LOG_LEVEL", "LOG_LEVEL") or "INFO",
        )

    @classmethod
    def from_env_file(cls, path: str) -> "RAGConfig":
        """**显式**地从 .env 文件构造配置（识别 FINRAG_* 与旧键）。"""
        values: dict[str, str] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    values[k.strip()] = v.strip().strip('"').strip("'")
        return cls.from_mapping(values)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """**显式**从进程环境变量构造配置。"""
        return cls.from_mapping(dict(os.environ))


def load_config(path: str | None = None) -> RAGConfig:
    """显式加载配置（仅应用/示例显式调用，库核心不会自动触发）。

    优先级：显式 ``path`` > 环境变量 ``FINRAG_ENV_FILE`` > 进程环境变量 > 默认值。
    """
    if path is None:
        path = os.environ.get("FINRAG_ENV_FILE")
    if path and os.path.exists(path):
        return RAGConfig.from_env_file(path)
    return RAGConfig.from_env()
