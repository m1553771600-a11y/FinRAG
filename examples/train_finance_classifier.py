"""金融意图分类器微调示例。

用法::

    python examples/train_finance_classifier.py \
        --data examples/finance_classifier_data.jsonl \
        --base-model sentence-transformers/all-MiniLM-L6-v2 \
        --output ./models/classifier/finance

依赖: ``finrag[classifier]`` + ``datasets`` + ``scikit-learn``（训练专用，推理只需 transformers）。
训练完成后用 ``QueryClassifier(output_dir)`` 加载推理。
"""

import argparse

from finrag import QueryClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="金融意图分类器微调")
    parser.add_argument("--data", required=True, help="训练数据 JSONL")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", default="./models/classifier/finance")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    QueryClassifier.train(
        args.data,
        args.output,
        base_model_path=args.base_model,
        epochs=args.epochs,
    )
    print(f"✅ 微调完成，模型保存到：{args.output}")
    print(f"   推理：from finrag import QueryClassifier; qc = QueryClassifier('{args.output}')")


if __name__ == "__main__":
    main()
