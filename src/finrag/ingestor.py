"""finrag.ingestor —— 文档加载/分块/嵌入/入库（建库 pipeline）。

对应旧 scripts/build_knowledge_base.py + init_milvus.py。
依赖: ``finrag[ingest]``（langchain loaders/splitters）+ ``finrag[embedding]``。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .config import RAGConfig
from .types import DocumentChunk, Embedder, VectorStore, ChatLLM, generate_chunk_id
from .errors import InvalidInputError
from ._logging import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 12
MAX_CONTEXT_CONCURRENCY = 5

CONTEXTUAL_CHUNK_PROMPT = """\
<document>
{document_text}
</document>

以下是需要在整个文档中定位的 chunk：
<chunk>
{chunk_content}
</chunk>

请用一句简洁的中文，描述这段内容在整个文档中的位置和作用，以便改善检索效果。
只输出这一句描述，不要加任何前缀或标签。"""


class Ingestor:
    """文档入库 pipeline：读取 -> 分块 -> (可选 Contextual RAG) -> 嵌入 -> 写入。"""

    def __init__(
        self,
        config: RAGConfig,
        *,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm: ChatLLM | None = None,
    ):
        self._config = config
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._md_header_splitter = None
        self._char_splitter = None

    # ── 惰性构造 ────────────────────────────────────────────────

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            from .embedder import BGE3Embedder

            if not self._config.embedding_model_path:
                raise InvalidInputError("未提供 embedding_model_path，也未注入 embedder")
            self._embedder = BGE3Embedder(self._config.embedding_model_path)
        return self._embedder

    def _get_store(self) -> VectorStore:
        if self._store is None:
            from .vector_store import MilvusStore

            self._store = MilvusStore(self._config.milvus_uri, self._config.collection_name)
        return self._store

    def _get_llm(self) -> ChatLLM:
        if self._llm is None:
            from .llm import OpenAICompatLLM

            if not self._config.llm_api_key:
                raise InvalidInputError("未提供 llm_api_key，也未注入 llm")
            self._llm = OpenAICompatLLM(
                self._config.llm_api_key,
                base_url=self._config.llm_base_url,
                model=self._config.llm_model,
            )
        return self._llm

    def _get_splitters(self):
        if self._char_splitter is None:
            from langchain_text_splitters import (
                MarkdownHeaderTextSplitter,
                RecursiveCharacterTextSplitter,
            )

            self._md_header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")],
                strip_headers=False,
            )
            self._char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", "，", " ", ""],
            )
        return self._md_header_splitter, self._char_splitter

    # ── 文档加载 ────────────────────────────────────────────────

    @staticmethod
    def load_document(file_path: str) -> list:
        """统一文档加载入口，根据扩展名选择 Loader（.pdf / .md / .markdown）。"""
        from langchain_community.document_loaders import PyPDFLoader, TextLoader

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")
        ext = path.suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(file_path).load()
        if ext in (".md", ".markdown"):
            return TextLoader(file_path, encoding="utf-8").load()
        raise ValueError(f"不支持的文件类型：{ext}\n当前支持：.pdf / .md / .markdown")

    # ── 分块 ────────────────────────────────────────────────────

    def split_documents(self, docs: list, file_path: str) -> list:
        """统一分块入口，根据文件类型选择策略。"""
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self._split_pdf(docs)
        if ext in (".md", ".markdown"):
            return self._split_markdown(docs)
        raise ValueError(f"不支持的文件类型：{ext}")

    def _split_pdf(self, pages: list) -> list:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        non_empty = [p for p in pages if len(p.page_content.strip()) > 20]
        _, char_splitter = self._get_splitters()
        chunks = char_splitter.split_documents(non_empty)
        for chunk in chunks:
            filename = Path(chunk.metadata.get("source", "未知文件")).stem
            page_num = chunk.metadata.get("page", 0) + 1
            chunk.metadata["source_name"] = f"{filename} 第{page_num}页"
        return chunks

    def _split_markdown(self, docs: list, chunk_size: int = 512, chunk_overlap: int = 100) -> list:
        from langchain_text_splitters import MarkdownTextSplitter

        header_splitter, _ = self._get_splitters()
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        header_chunks: list = []
        for doc in docs:
            sections = header_splitter.split_text(doc.page_content)
            source_path = doc.metadata.get("source", "")
            for section in sections:
                section.metadata["source"] = source_path
            header_chunks.extend(sections)

        final_chunks = splitter.split_documents(header_chunks)
        for chunk in final_chunks:
            source_path = chunk.metadata.get("source", "")
            filename = Path(source_path).stem if source_path else "未知文件"
            parts = [
                chunk.metadata.get("H1", ""),
                chunk.metadata.get("H2", ""),
                chunk.metadata.get("H3", ""),
                chunk.metadata.get("H4", ""),
            ]
            parts = [p for p in parts if p]
            chunk.metadata["source_name"] = (
                f"{filename} > {' > '.join(parts)}" if parts else filename
            )
        return final_chunks

    # ── 嵌入 ────────────────────────────────────────────────────

    def embed_chunks(
        self,
        chunks: list,
        course_id: str,
        document_id: str,
        tenant_id: str = "tenant_default",
        version: str = "1.0",
    ) -> list[DocumentChunk]:
        """对 chunk 列表做 BGE-M3 嵌入，返回 DocumentChunk 列表。"""
        embedder = self._get_embedder()
        all_chunks: list[DocumentChunk] = []
        total = len(chunks)
        for batch_start in range(0, total, BATCH_SIZE):
            batch = chunks[batch_start: batch_start + BATCH_SIZE]
            texts = [c.page_content for c in batch]
            dense_vecs, sparse_vecs = embedder.encode(texts, batch_size=BATCH_SIZE)
            for i, (chunk, dense, sparse) in enumerate(zip(batch, dense_vecs, sparse_vecs)):
                global_index = batch_start + i
                all_chunks.append(DocumentChunk(
                    id=generate_chunk_id(chunk.page_content, document_id, global_index),
                    content=chunk.page_content,
                    embedding=dense,
                    sparse_embedding=sparse,
                    course_id=course_id,
                    document_id=document_id,
                    source_name=chunk.metadata.get("source_name", ""),
                    chunk_type=chunk.metadata.get("chunk_type", "text"),
                    chunk_index=global_index,
                    version=version,
                    tenant_id=tenant_id,
                ))
        return all_chunks

    # ── Contextual RAG ──────────────────────────────────────────

    async def _generate_chunk_context(self, llm, document_text, chunk_content, semaphore) -> str:
        async with semaphore:
            try:
                from langchain_core.messages import HumanMessage

                prompt = CONTEXTUAL_CHUNK_PROMPT.format(
                    document_text=document_text, chunk_content=chunk_content
                )
                resp = await llm.ainvoke([HumanMessage(content=prompt)])
                text = resp.text if hasattr(resp, "text") and not callable(resp.text) else str(resp.content)
                return text.strip()
            except Exception as e:  # noqa: BLE001
                logger.warning("ingestor.context_failed", error=str(e))
                return ""

    async def add_context(self, chunks: list, docs: list, concurrency: int = MAX_CONTEXT_CONCURRENCY) -> list:
        """Contextual RAG：并发为每个 chunk 生成一句上下文描述并拼接到前方。"""
        llm = self._get_llm()
        full_doc_text = "\n\n".join(d.page_content for d in docs)[:8000]
        semaphore = asyncio.Semaphore(concurrency)
        contexts = await asyncio.gather(*[
            self._generate_chunk_context(llm, full_doc_text, c.page_content, semaphore)
            for c in chunks
        ])
        enriched = 0
        for chunk, ctx in zip(chunks, contexts):
            if ctx:
                chunk.page_content = f"{ctx}\n\n{chunk.page_content}"
                enriched += 1
        logger.info("ingestor.context_enriched", enriched=enriched, total=len(chunks))
        return chunks

    # ── 写入 ────────────────────────────────────────────────────

    def write(self, chunks: list[DocumentChunk]) -> int:
        """将 DocumentChunk 列表写入向量库，返回写入条数。"""
        if not chunks:
            return 0
        return self._get_store().upsert(chunks)

    def ensure_collection(self, *, drop_existing: bool = False) -> None:
        """构建 collection schema + 索引。"""
        store = self._get_store()
        if hasattr(store, "ensure_collection"):
            store.ensure_collection(drop_existing=drop_existing)

    # ── 完整 pipeline ──────────────────────────────────────────

    async def build(
        self,
        file_path: str,
        course_id: str,
        document_id: str,
        tenant_id: str = "tenant_default",
        version: str = "1.0",
        use_context: bool = True,
    ) -> int:
        """建库完整流水线：读取 -> 分块 -> (Contextual RAG) -> 嵌入 -> 写入。返回 chunk 数。"""
        docs = self.load_document(file_path)
        chunks = self.split_documents(docs, file_path)
        if use_context and chunks:
            chunks = await self.add_context(chunks, docs)
        doc_chunks = self.embed_chunks(chunks, course_id, document_id, tenant_id=tenant_id, version=version)
        self.write(doc_chunks)
        return len(doc_chunks)
