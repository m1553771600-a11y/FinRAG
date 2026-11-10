"""最小示例：导入金融文档到 Milvus。

配置来源同 finance_query.py（D:\\RAG\\.env 或 FINRAG_* 环境变量）。

用法::

    python examples/ingest_finance_doc.py

默认导入 examples/finance_sample.md，course_id = "finance"。
"""

import asyncio
import uuid
from pathlib import Path

from finrag import load_config, Ingestor

_HERE = Path(__file__).resolve().parent


async def main() -> None:
    cfg = load_config(str(_HERE.parent / ".env"))

    ing = Ingestor(cfg)
    # 首次导入可改为 drop_existing=True 重建集合（会清空数据，谨慎）
    ing.ensure_collection(drop_existing=False)

    doc_id = str(uuid.uuid4())
    n = await ing.build(
        str(_HERE / "finance_sample.md"),
        course_id="finance",
        document_id=doc_id,
        tenant_id="tenant_default",
        version="1.0",
        use_context=False,  # 关闭 Contextual RAG，避免消耗 LLM 配额
    )
    print(f"导入完成：{n} 个 chunk，document_id={doc_id}")


if __name__ == "__main__":
    asyncio.run(main())
