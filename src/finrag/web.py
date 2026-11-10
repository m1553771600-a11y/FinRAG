"""finrag.web —— Web 搜索兜底（可选）。

低置信度 / 空检索时，可用 Web 搜索补充互联网信息。
依赖: ``finrag[web]``（duckduckgo-search）；Tavily 走 httpx。
"""

from __future__ import annotations

from typing import Protocol

from ._logging import get_logger

logger = get_logger(__name__)


class WebSearch(Protocol):
    """Web 搜索协议。返回 [{"title","url","snippet"}, ...]。"""

    async def search(self, query: str, max_results: int = 5) -> list[dict]: ...


class DuckDuckGoSearch:
    """DuckDuckGo 搜索（免费，无需 API key）。"""

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        import asyncio

        from duckduckgo_search import DDGS

        def _sync() -> list[dict]:
            out: list[dict] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    out.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:500],
                    })
            return out

        return await asyncio.to_thread(_sync)


class TavilySearch:
    """Tavily 搜索（结构化结果，需 API key）。"""

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        import httpx

        async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
            resp.raise_for_status()
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:500],
            }
            for r in resp.json().get("results", [])
        ]


async def search_web(
    query: str,
    max_results: int = 5,
    *,
    api_key: str | None = None,
) -> list[dict]:
    """优先 Tavily（若提供 key），否则 DuckDuckGo。任一失败则静默降级。"""
    if api_key:
        try:
            results = await TavilySearch(api_key).search(query, max_results)
            logger.info("web.tavily_done", hits=len(results))
            return results
        except Exception as e:  # noqa: BLE001
            logger.warning("web.tavily_failed", error=str(e))

    try:
        results = await DuckDuckGoSearch().search(query, max_results)
        logger.info("web.ddgs_done", hits=len(results))
        return results
    except Exception as e:  # noqa: BLE001
        logger.warning("web.ddgs_failed", error=str(e))
        return []
