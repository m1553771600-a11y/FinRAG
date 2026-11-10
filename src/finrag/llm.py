"""finrag.llm —— OpenAI 兼容 LLM 默认实现。

依赖: ``finrag[llm]``（langchain-openai + langchain-core + httpx）。
行为与旧 llm_factory 一致：trust_env=False 绕过系统代理、temperature 可控。
"""

from __future__ import annotations

from typing import Any


class OpenAICompatLLM:
    """OpenAI 兼容接口的对话模型（DeepSeek / OpenAI / 任意兼容端点）。

    用法::

        llm = OpenAICompatLLM(api_key="...", base_url="https://api.deepseek.com", model="deepseek-chat")
        resp = await llm.ainvoke(messages)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        *,
        timeout: float = 120.0,
        connect_timeout: float = 15.0,
        extra_body: dict | None = None,
    ):
        import httpx
        from langchain_openai import ChatOpenAI

        # trust_env=False：忽略系统代理与 HTTP(S)_PROXY 环境变量（与旧实现一致）
        http_client = httpx.Client(
            trust_env=False, timeout=httpx.Timeout(timeout, connect=connect_timeout)
        )
        async_client = httpx.AsyncClient(
            trust_env=False, timeout=httpx.Timeout(timeout, connect=connect_timeout)
        )

        kwargs: dict[str, Any] = dict(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_retries=0,
            http_client=http_client,
            http_async_client=async_client,
        )
        if base_url:
            kwargs["base_url"] = base_url
        if extra_body:
            kwargs["extra_body"] = extra_body

        self._model = ChatOpenAI(**kwargs)

    def invoke(self, messages: list) -> object:
        return self._model.invoke(messages)

    async def ainvoke(self, messages: list) -> object:
        return await self._model.ainvoke(messages)
