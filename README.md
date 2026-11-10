# FinSage（金融智者）

面向金融领域的 RAG 智能问答库。先检索可信资料，再让大模型基于检索结果作答并标注出处，把「能回答」升级成「答得准、有出处」。

- **检索**：BGE-M3 双向量（dense + sparse）→ Milvus Hybrid 检索（WeightedRanker）→ BGE-Reranker 精排
- **入库**：文档加载 / 分块 / 嵌入 / 写入，支持 Contextual RAG
- **问答**：拼 prompt → LLM → 回答，prompt 可注入；空检索时可 Web 搜索兜底
- **意图分类**：QueryClassifier 二分类（可选，支持微调）
- **设计约束**：库内部**绝不**读全局变量、环境变量或任何隐式配置；外部依赖（embedding / 重排 / 向量库 / LLM / Web 搜索）全部通过 Protocol 注入，可整体替换

> Python 包名：`finrag`，导入方式 `import finrag`。

## 安装

```bash
# 最小安装（仅向量库）
pip install -e D:\RAG

# 金融 RAG 全家桶
pip install -e "D:\RAG[all]"
```

可选 extras：`embedding` / `reranker` / `llm` / `ingest` / `classifier` / `web` / `all`。

## 配置（自包含）

库自身不读环境变量。示例/应用通过 `load_config()` **显式**加载配置：

```bash
# 复制模板并填写
cp .env.example .env
```

```python
from finrag import load_config, Retriever

cfg = load_config("D:/RAG/.env")   # 缺省回退到 FINRAG_* 环境变量
retriever = Retriever(cfg)
docs, confidence = retriever.retrieve("什么是市盈率？", tenant_id="tenant_default")
```

`.env` 使用 `FINRAG_*` 键：

```ini
FINRAG_MILVUS_URI=http://localhost:19530
FINRAG_COLLECTION_NAME=finance_knowledge
FINRAG_EMBEDDING_MODEL_PATH=./models/embedding/bge-m3
FINRAG_RERANKER_MODEL_PATH=./models/reranker/bge-reranker-large
FINRAG_LLM_API_KEY=...
FINRAG_LLM_BASE_URL=https://api.deepseek.com
FINRAG_LLM_MODEL=deepseek-chat
```

> 模型权重（BGE-M3 / BGE-Reranker 等）属大体积二进制，不随库分发；请把 `FINRAG_*_MODEL_PATH` 指向你自己的权重目录。

## 快速开始（金融 query）

```bash
# 1. 启动 Milvus
# 2. 导入文档（BGE-M3 嵌入，不走 LLM）
python examples/ingest_finance_doc.py
# 3. 检索 + 生成
python examples/finance_query.py
```

```python
from finrag import load_config, Retriever, AnswerGenerator, OpenAICompatLLM

cfg = load_config("D:/RAG/.env")
docs, confidence = Retriever(cfg).retrieve("什么是市盈率？如何判断估值高低？", tenant_id="tenant_default")

llm = OpenAICompatLLM(cfg.llm_api_key, base_url=cfg.llm_base_url, model=cfg.llm_model)
answer = await AnswerGenerator(llm).answer("什么是市盈率？", docs)
print(answer.content, answer.sources, answer.mode, answer.confidence)
```

## 公共 API

| 组件 | 说明 |
|---|---|
| `RAGConfig` / `load_config()` | 显式配置与加载辅助 |
| `Retriever(config).retrieve(query, tenant_id, course_id, recall_top_k, rerank_top_k)` | 检索 pipeline → `(list[RankedDocument], confidence)` |
| `Ingestor(config)` | `load_document / split_documents / embed_chunks / add_context / write / ensure_collection / build` |
| `AnswerGenerator(llm).answer(query, ranked_docs, ...)` | 拼 prompt → LLM → `Answer`（含 Web 兜底） |
| `QueryClassifier(model_path)` | 意图二分类 `classify()` + 类方法 `train()` |
| `OpenAICompatLLM` | OpenAI 兼容 LLM 默认实现 |
| `DuckDuckGoSearch` / `TavilySearch` / `search_web` | Web 搜索实现 |
| `DocumentChunk` / `RankedDocument` / `generate_chunk_id` | 数据结构与工具 |
| `finrag.domains.finance` | 金融关键词 / 系统/RAG/HyDE/MultiQuery prompt / 阈值 |

## 如何替换 embedding / 重排 / 向量库 / LLM / Web 搜索

依赖通过 `typing.Protocol` 注入，满足协议即可替换：

```python
from finrag import Retriever, RAGConfig
from finrag.types import DocumentChunk, RankedDocument

class MyEmbedder:                          # 替换 embedding
    def encode_query(self, text): return [0.0] * 1024, {}
    def encode(self, texts, batch_size=12): return [[0.0] * 1024] * len(texts), [{}] * len(texts)

class MyStore:                             # 替换向量库
    def upsert(self, chunks): return len(chunks)
    def delete(self, document_id): pass
    def search(self, dense, sparse, top_k, filters): return []
    def build_filter(self, tenant_id, course_id=None): return f'tenant_id == "{tenant_id}"'

class MyReranker:                          # 替换重排
    def rerank(self, query, documents, top_k=3):
        ranked = [RankedDocument(d["content"], d["score"], i, d.get("metadata", {}))
                  for i, d in enumerate(documents[:top_k])]
        return ranked, (ranked[0].score if ranked else 0.0)

retriever = Retriever(cfg, embedder=MyEmbedder(), store=MyStore(), reranker=MyReranker())
```

LLM 实现 `invoke` / `ainvoke`（接收 langchain 消息列表）即可；Web 搜索实现 `async def search(query, max_results)` 即可。

## Web 搜索兜底

空检索 / 低置信度时，可注入 `web_search` 让回答联网补充：

```python
from finrag import AnswerGenerator, DuckDuckGoSearch

gen = AnswerGenerator(llm, web_search=DuckDuckGoSearch())
answer = await gen.answer("最新 CPI 是多少？", [])   # 自动联网，mode="web_augmented"
```

## 意图分类器微调

```bash
python examples/train_finance_classifier.py \
    --data examples/finance_classifier_data.jsonl \
    --base-model sentence-transformers/all-MiniLM-L6-v2 \
    --output ./models/classifier/finance
```

训练完成后加载推理：`QueryClassifier("./models/classifier/finance").classify(text)`。

## 失败 / 空检索约定

| 场景 | 约定 |
|---|---|
| 检索命中 0 条 | `retrieve()` 返回 `([], 0.0)`，不抛异常 |
| 精排输入为空 | `rerank()` 返回 `([], 0.0)` |
| 向量库连接失败 | 抛 `MilvusConnectionError` |
| LLM 调用失败 | 抛 `LLMAPIError`（由调用方决定降级） |
| 文档不存在 / 类型不支持 | 抛 `FileNotFoundError` / `ValueError` |
| 入库空列表 | 返回 `0` |

## 领域定制

库核心（`retriever` / `ingestor` / `embedder` / `reranker` / `vector_store` / `answer`）**不含任何领域词**。默认领域内容集中在 `finrag/domains/finance.py`（系统/RAG/HyDE/MultiQuery prompt + 金融关键词 + 阈值），可通过 `AnswerGenerator` 的 `system_prompt` / `rag_prompt_template` / `direct_prompt_template` 参数整体替换，或新增自己的领域模块。

## 版本

`0.1.0`。0.x 允许 breaking change（CHANGELOG 记录）；1.0 起冻结公共 API。
