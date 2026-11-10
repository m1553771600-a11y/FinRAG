from finrag.web import DuckDuckGoSearch, TavilySearch, WebSearch
from finrag.answer import AnswerGenerator


def test_web_protocol_and_impls():
    assert WebSearch is not None
    assert DuckDuckGoSearch() is not None
    t = TavilySearch("k")
    assert t._api_key == "k"


def test_build_web_context():
    results = [
        {"title": "t1", "url": "http://a", "snippet": "s1"},
        {"title": "t2", "url": "http://b", "snippet": "s2"},
    ]
    ctx, sources = AnswerGenerator.build_web_context(results)
    assert "http://a" in ctx
    assert "http://b" in ctx
    assert sources == ["http://a", "http://b"]
