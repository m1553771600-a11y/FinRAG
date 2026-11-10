from finrag.domains import finance


def test_finance_keywords_present():
    assert "市盈率" in finance.FINANCE_SPECIALIZED_KEYWORDS
    assert "股票" in finance.FINANCE_SPECIALIZED_KEYWORDS
    assert "基金" in finance.FINANCE_SPECIALIZED_KEYWORDS


def test_no_editing_keywords():
    # 剪辑领域词必须已全部移除
    for w in ("剪辑", "剪映", "premiere", "达芬奇", "davinci", "字幕", "调色", "转场", "卡点", "蒙版"):
        assert w not in finance.FINANCE_SPECIALIZED_KEYWORDS


def test_prompts_reference_finance():
    assert "金融" in finance.FINANCE_SYSTEM_PROMPT
    assert "{context}" in finance.FINANCE_RAG_ANSWER_PROMPT
    assert "{query}" in finance.FINANCE_RAG_ANSWER_PROMPT
    assert "{query}" in finance.FINANCE_DIRECT_ANSWER_PROMPT
