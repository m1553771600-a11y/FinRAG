import asyncio

from finrag.answer import AnswerGenerator
from finrag.types import RankedDocument


class _Resp:
    text = "回答内容"
    content = "回答内容"


class FakeLLM:
    async def ainvoke(self, messages):
        return _Resp()

    def invoke(self, messages):
        return _Resp()


def test_answer_rag():
    docs = [RankedDocument("市盈率是股价除以每股收益", 0.9, 0, {"source_name": "研报"})]
    gen = AnswerGenerator(FakeLLM())
    ans = asyncio.run(gen.answer("什么是市盈率", docs))
    assert ans.mode == "rag"
    assert "参考来源" in ans.content
    assert ans.sources == ["研报"]
    assert ans.confidence == 0.9


def test_answer_direct():
    gen = AnswerGenerator(FakeLLM())
    ans = asyncio.run(gen.answer("你好", []))
    assert ans.mode == "direct"
    assert ans.confidence == 0.0
    assert ans.sources == []


def test_custom_prompt():
    gen = AnswerGenerator(FakeLLM(), rag_prompt_template="【自定义】{context} / {query}")
    docs = [RankedDocument("x", 0.9, 0, {"source_name": "s"})]
    ans = asyncio.run(gen.answer("q", docs))
    assert ans.mode == "rag"


class FakeWebSearch:
    async def search(self, query, max_results=5):
        return [{"title": "t", "url": "http://u", "snippet": "s"}]


def test_answer_web_augmented():
    gen = AnswerGenerator(FakeLLM(), web_search=FakeWebSearch())
    ans = asyncio.run(gen.answer("最新 CPI 数据", []))
    assert ans.mode == "web_augmented"
    assert ans.sources == ["http://u"]
