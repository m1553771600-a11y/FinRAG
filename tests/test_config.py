from finrag import RAGConfig, load_config


def test_defaults():
    c = RAGConfig()
    assert c.collection_name == "finance_knowledge"
    assert c.recall_top_k == 10
    assert c.rerank_top_k == 3
    assert c.confidence_threshold == 0.75


def test_frozen_and_replace():
    import dataclasses

    c = RAGConfig()
    c2 = dataclasses.replace(c, rerank_top_k=5)
    assert c2.rerank_top_k == 5
    assert c.rerank_top_k == 3


def test_from_env_file(tmp_path):
    p = tmp_path / ".env.local"
    p.write_text(
        "MILVUS_HOST=localhost\n"
        "MILVUS_PORT=19532\n"
        "DEEPSEEK_API_KEY=abc\n"
        "DEEPSEEK_BASE_URL=https://api.deepseek.com/\n"
        "DEEPSEEK_MODEL_CHAT=deepseek-v4-flash\n"
        "LOG_LEVEL=INFO\n",
        encoding="utf-8",
    )
    c = RAGConfig.from_env_file(str(p))
    assert c.milvus_uri == "http://localhost:19532"
    assert c.llm_api_key == "abc"
    assert c.llm_base_url == "https://api.deepseek.com/"
    assert c.llm_model == "deepseek-v4-flash"


def test_from_mapping_finrag_keys():
    c = RAGConfig.from_mapping({
        "FINRAG_MILVUS_URI": "http://h:1",
        "FINRAG_COLLECTION_NAME": "c",
        "FINRAG_EMBEDDING_MODEL_PATH": "e",
        "FINRAG_LLM_API_KEY": "k",
        "FINRAG_LLM_MODEL": "m",
    })
    assert c.milvus_uri == "http://h:1"
    assert c.collection_name == "c"
    assert c.embedding_model_path == "e"
    assert c.llm_api_key == "k"
    assert c.llm_model == "m"


def test_from_mapping_legacy_keys():
    c = RAGConfig.from_mapping({"MILVUS_HOST": "h", "MILVUS_PORT": "123", "DEEPSEEK_API_KEY": "k"})
    assert c.milvus_uri == "http://h:123"
    assert c.llm_api_key == "k"


def test_load_config_fallback_to_env(monkeypatch):
    monkeypatch.setenv("FINRAG_MILVUS_URI", "http://x:9")
    c = load_config("/nonexistent/.env")  # 文件不存在 -> 回退环境变量
    assert c.milvus_uri == "http://x:9"
