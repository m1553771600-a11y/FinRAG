"""finrag.reranker —— BGE-Reranker 精排（默认实现）。

依赖: ``finrag[reranker]``（sentence-transformers + transformers + torch）。
"""

from __future__ import annotations

from typing import Optional

from .types import RankedDocument
from ._logging import get_logger

logger = get_logger(__name__)


class BGERerankerImpl:
    """BGE-Reranker CrossEncoder 精排，输出 [0, 1] 置信度。

    用法::

        reranker = BGERerankerImpl("/path/to/bge-reranker-large")
        docs, confidence = reranker.rerank(query, candidates, top_k=3)
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        max_input_chars: int = 512,
        disable_accelerate_meta: bool = True,
    ):
        import os

        import torch
        from sentence_transformers import CrossEncoder

        if disable_accelerate_meta:
            # 与旧实现一致：禁用 accelerate 的 meta device 探测，避免加载异常
            os.environ["ACCELERATE_USE_META_DEVICE"] = "0"

        use_local = (
            bool(model_path)
            and os.path.isdir(model_path)
            and any(f.endswith((".bin", ".safetensors", ".json")) for f in os.listdir(model_path))
        )
        model_id = model_path if use_local else "BAAI/bge-reranker-v2-m3"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("reranker.loading", model_id=model_id, device=device)
        self._model = CrossEncoder(model_id, device=device, max_length=512)
        self._max_input_chars = max_input_chars
        logger.info("reranker.loaded", model_id=model_id)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3,
    ) -> tuple[list[RankedDocument], float]:
        """精排并返回置信度。返回 (ranked_docs, confidence)。"""
        if not documents:
            return [], 0.0

        pairs = [
            (query, (doc.get("content") or "")[: self._max_input_chars])
            for doc in documents
        ]
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(
            [
                RankedDocument(
                    content=documents[i].get("content", ""),
                    score=scores[i],
                    original_index=i,
                    metadata=documents[i].get("metadata", {}),
                )
                for i in range(len(documents))
            ],
            key=lambda x: x.score,
            reverse=True,
        )
        top_results = ranked[:top_k]
        confidence = top_results[0].score if top_results else 0.0

        logger.info(
            "reranker.done",
            candidates=len(documents),
            top_k=top_k,
            confidence=round(confidence, 4),
        )
        return top_results, confidence
