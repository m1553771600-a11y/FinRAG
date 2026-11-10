"""finrag.embedder —— BGE-M3 双向量嵌入（默认实现）。

依赖: ``finrag[embedding]``（FlagEmbedding + torch）。
所有重依赖均在构造时惰性导入，``import finrag`` 不触发加载。
"""

from __future__ import annotations

from ._logging import get_logger

logger = get_logger(__name__)


class BGE3Embedder:
    """BGE-M3 本地嵌入模型。一次推理同时输出 dense + sparse 双向量。

    用法::

        embedder = BGE3Embedder("/path/to/bge-m3")
        dense, sparse = embedder.encode_query("什么是市盈率？")
    """

    def __init__(self, model_path: str):
        # 兼容性补丁：transformers 未来版本移除 is_torch_fx_available 时兜底
        import importlib.util as _ilu
        from transformers.utils import import_utils as _tf_iu

        if not hasattr(_tf_iu, "is_torch_fx_available"):
            _tf_iu.is_torch_fx_available = lambda: _ilu.find_spec("torch.fx") is not None

        import torch
        from FlagEmbedding import BGEM3FlagModel

        # fp16 仅在 CUDA 上启用（MPS 在 BGE-M3 attention 上会触发 LLVM ERROR）
        use_fp16 = torch.cuda.is_available()

        logger.info("bge_m3.loading", model_path=model_path)
        self._model = BGEM3FlagModel(model_name_or_path=model_path, use_fp16=use_fp16)
        logger.info("bge_m3.loaded", use_fp16=use_fp16)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
    ) -> tuple[list[list[float]], list[dict]]:
        """批量编码，返回 (dense_vecs, sparse_vecs)。"""
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = output["dense_vecs"].tolist()
        # sparse: numpy.float16 -> Python float（msgpack 序列化兼容）
        sparse_vecs = [{int(k): float(v) for k, v in d.items()} for d in output["lexical_weights"]]
        return dense_vecs, sparse_vecs

    def encode_query(self, text: str) -> tuple[list[float], dict]:
        """编码单条查询，返回 (dense_vec, sparse_vec)。"""
        dense_list, sparse_list = self.encode([text], batch_size=1)
        return dense_list[0], sparse_list[0]
