"""最小示例：金融知识库检索 + 问答（finrag）。

配置来源（二选一，库本身不隐式读环境）：
  1. D:\\RAG\\.env —— 复制 .env.example 为 .env 并填写
  2. 进程环境变量 FINRAG_*（如 FINRAG_MILVUS_URI / FINRAG_LLM_API_KEY 等）

用法（先启动 Milvus 并运行 ingest_finance_doc.py 导入文档）::

    python examples/finance_query.py
"""

import asyncio
from pathlib import Path

from finrag import load_config, Retriever, AnswerGenerator, OpenAICompatLLM

_HERE = Path(__file__).resolve().parent


async def main() -> None:
    cfg = load_config(str(_HERE.parent / ".env"))  # 缺省回退到环境变量

    query = "什么是市盈率？如何判断估值高低？"

    # ── 检索 ──
    retriever = Retriever(cfg)
    docs, confidence = retriever.retrieve(query, tenant_id="tenant_default")
    print(f"检索到 {len(docs)} 条，置信度 {confidence:.4f}")
    for d in docs:
        print(f"  [{d.score:.4f}] {d.metadata.get('source_name', '')}")
        print(f"      {d.content[:80]}...")

    # ── 生成 ──
    llm = OpenAICompatLLM(
        cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        extra_body={"thinking": {"type": "disabled"}},
    )
    gen = AnswerGenerator(llm)
    answer = await gen.answer(query, docs, confidence=confidence)
    print("\n===== 回答 =====\n")
    print(answer.content)
    print("\n===== 元数据 =====")
    print(f"mode={answer.mode} sources={answer.sources} confidence={answer.confidence:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
