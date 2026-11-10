"""finrag.classifier —— 意图二分类器（可选组件）。

领域无关的 general/specialized 二分类；模型路径与阈值由调用方显式提供。
依赖: ``finrag[classifier]``（transformers）。
"""

from __future__ import annotations

from typing import Optional

from ._logging import get_logger

logger = get_logger(__name__)

LABEL2ID = {"general": 0, "specialized": 1}
ID2LABEL = {0: "general", 1: "specialized"}


class QueryClassifier:
    """Query 二分类器：general / specialized。

    用法::

        qc = QueryClassifier("/path/to/classifier")
        label, confidence = qc.classify("什么是市盈率？")
    """

    def __init__(
        self,
        model_path: str,
        *,
        confidence_threshold: float = 0.85,
        device: Optional[int] = None,
    ):
        import torch
        from transformers import pipeline as hf_pipeline

        self._confidence_threshold = confidence_threshold
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self._pipeline = hf_pipeline(
            task="text-classification",
            model=model_path,
            device=device,
            top_k=None,
            truncation=True,
            max_length=128,
        )
        logger.info("query_classifier.loaded", model_id=model_path)

    def classify(self, text: str) -> tuple[str, float]:
        """返回 (label, confidence)，label 为 "general" / "specialized"。"""
        raw: list[dict] = self._pipeline(text)[0]

        general_score: Optional[float] = None
        for item in raw:
            lbl = item["label"].lower()
            if lbl in ("general", "label_0"):
                general_score = item["score"]
                break

        if general_score is None:
            logger.warning("query_classifier.unexpected_labels", labels=[x["label"] for x in raw])
            return "specialized", 0.5

        if general_score >= self._confidence_threshold:
            return "general", general_score
        return "specialized", 1.0 - general_score

    @classmethod
    def train(
        cls,
        data_path: str,
        output_dir: str,
        base_model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        epochs: int = 8,
        batch_size: int = 64,
        lr: float = 2e-5,
        max_length: int = 128,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """微调基座模型并保存到 output_dir（领域无关，金融训练数据需自行准备）。

        Args:
            data_path:      训练数据 JSONL（每行 {"text":..., "label":"general"/"specialized"}）
            output_dir:     微调模型保存目录
            base_model_path: 基座模型（本地路径或 HuggingFace 模型名）
        """
        import json  # noqa: F401
        import random  # noqa: F401
        from pathlib import Path

        import numpy as np
        import torch
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        set_seed(seed)

        rows = cls._load_jsonl(data_path)
        train_rows, val_rows, test_rows = cls._stratified_split(
            rows, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )

        model_id = base_model_path
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,
        )

        def to_dataset(rows_: list[dict]) -> Dataset:
            return Dataset.from_dict({
                "text": [r["text"] for r in rows_],
                "label": [LABEL2ID[r["label"]] for r in rows_],
            })

        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=max_length)

        train_ds = to_dataset(train_rows).map(tokenize, batched=True, remove_columns=["text"])
        val_ds = to_dataset(val_rows).map(tokenize, batched=True, remove_columns=["text"])
        test_ds = to_dataset(test_rows).map(tokenize, batched=True, remove_columns=["text"])

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, preds)
            _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
            return {"accuracy": float(acc), "f1_macro": float(f1)}

        use_cuda = torch.cuda.is_available()
        checkpoint_dir = str(Path(output_dir) / "_checkpoints")
        train_args = TrainingArguments(
            output_dir=checkpoint_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_total_limit=1,
            logging_steps=20,
            fp16=use_cuda,
            report_to="none",
            seed=seed,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("query_classifier.saved", output_dir=output_dir)

    @staticmethod
    def _load_jsonl(path: str) -> list[dict]:
        import json

        rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = (obj.get("text") or "").strip()
                label = obj.get("label")
                if not text:
                    raise ValueError(f"第 {line_no} 行 text 为空")
                if label not in LABEL2ID:
                    raise ValueError(f"第 {line_no} 行 label 非法: {label!r}")
                rows.append({"text": text, "label": label})
        if not rows:
            raise ValueError(f"训练数据为空：{path}")
        return rows

    @staticmethod
    def _stratified_split(rows, val_ratio, test_ratio, seed):
        import random

        random.seed(seed)
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(row["label"], []).append(row)

        train_rows, val_rows, test_rows = [], [], []
        for group in buckets.values():
            random.shuffle(group)
            n = len(group)
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            test_rows.extend(group[:n_test])
            val_rows.extend(group[n_test:n_test + n_val])
            train_rows.extend(group[n_test + n_val:])
        random.shuffle(train_rows)
        return train_rows, val_rows, test_rows
