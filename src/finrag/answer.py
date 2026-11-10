"""finrag.answer —— 拼 prompt -> LLM -> 回答。

prompt 可注入；缺省使用金融领域 profile 的 prompt。
空检索 / 低置信度时可选 Web 搜索兜底。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import RankedDocument, ChatLLM
from .web import WebSearch
from ._logging import get_logger
from .domains.finance import (
    FINANCE_SYSTEM_PROMPT,
    FINANCE_RAG_ANSWER_PROMPT,
    FINANCE_DIRECT_ANSWER_PROMPT,
)

logger = get_logger(__name__)


@dataclass
class Answer:
    content: str
    sources: list[str]
    mode: str          # "rag" / "web_augmented" / "direct"
    confidence: float


def _message_text(msg) -> str:
    """统一获取消息文本（兼容 .text 与 .content）。"""
    if hasattr(msg, "text") and not callable(getattr(msg, "text", None)):
        return msg.text
    content = getattr(msg, "content", "")
    return content if isinstance(content, str) else str(content)


class AnswerGenerator:
    """基于检索结果生成回答（拼 prompt -> LLM -> 回答）。"""

    def __init__(
        self,
        llm: ChatLLM,
        *,
        system_prompt: str | None = None,
        rag_prompt_template: str | None = None,
        direct_prompt_template: str | None = None,
        include_sources: bool = True,
        web_search: WebSearch | None = None,
    ):
        self._llm = llm
        self._system_prompt = system_prompt or FINANCE_SYSTEM_PROMPT
        self._rag_prompt_template = rag_prompt_template or FINANCE_RAG_ANSWER_PROMPT
        self._direct_prompt_template = direct_prompt_template or FINANCE_DIRECT_ANSWER_PROMPT
        self._include_sources = include_sources
        self._web_search = web_search

    @staticmethod
    def build_context(ranked_docs: list[RankedDocument]) -> str:
        parts = []
        for i, doc in enumerate(ranked_docs, 1):
            parts.append(f"【参考{i}】\n{doc.content}")
        return "\n\n".join(parts)

    @staticmethod
    def build_web_context(results: list[dict]) -> tuple[str, list[str]]:
        """把 Web 搜索结果拼成 prompt 片段，返回 (context, sources)。"""
        snippets = "\n".join(
            f"  [{i + 1}] {r.get('title', '')}（{r.get('url', '')}）\n"
            f"      {r.get('snippet', '')[:300]}"
            for i, r in enumerate(results)
        )
        sources = [r.get("url", "") for r in results if r.get("url")]
        return snippets, sources

    async def answer(
        self,
        query: str,
        ranked_docs: list[RankedDocument],
        *,
        history: list | None = None,
        confidence: float | None = None,
        web_results: Optional[list[dict]] = None,
    ) -> Answer:
        """生成回答。

        Args:
            query:       用户问题
            ranked_docs: 精排后的文档（空 -> 走直答/Web 兜底）
            history:     历史消息列表（可选）
            confidence:  置信度（缺省取 Top-1 分数）
            web_results: 外部 Web 搜索结果（可选；未传且配置了 web_search 时自动搜索）
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        if confidence is None:
            confidence = ranked_docs[0].score if ranked_docs else 0.0

        if ranked_docs:
            context = self.build_context(ranked_docs)
            prompt = self._rag_prompt_template.format(context=context, query=query)
            mode = "rag"
            sources = [
                d.metadata.get("source_name", "")
                for d in ranked_docs
                if d.metadata.get("source_name")
            ]
            sources = list(dict.fromkeys(s for s in sources if s))
        else:
            # 空检索：优先 Web 兜底（若可用），否则 LLM 直答
            web_sources: list[str] = []
            if web_results is None and self._web_search is not None:
                try:
                    web_results = await self._web_search.search(query, max_results=3)
                except Exception as e:  # noqa: BLE001
                    logger.warning("answer.web_failed", error=str(e))
                    web_results = []

            if web_results:
                web_context, web_sources = self.build_web_context(web_results)
                prompt = self._direct_prompt_template.format(query=query) + (
                    f"\n\n【Web 搜索补充参考】\n{web_context}"
                )
                mode = "web_augmented"
                sources = web_sources
            else:
                prompt = self._direct_prompt_template.format(query=query)
                mode = "direct"
                sources = []

        messages: list = [SystemMessage(content=self._system_prompt)]
        for m in (history or []):
            messages.append(m)
        messages.append(HumanMessage(content=prompt))

        resp = await self._llm.ainvoke(messages)
        text = _message_text(resp).strip()

        if mode == "rag" and self._include_sources and sources:
            src_lines = "\n".join(f"  • {s}" for s in sources)
            text = f"{text}\n\n📚 **参考来源**\n{src_lines}"

        logger.info("answer.done", mode=mode, answer_length=len(text), sources=len(sources))
        return Answer(content=text, sources=sources, mode=mode, confidence=confidence)
