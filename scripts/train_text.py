"""文本 AIGC 检测模型训练脚本。

这个脚本属于后续“本地模型路线”，不是当前 API-first MVP 的必要步骤。
当前内容是训练 scaffold：
- 使用 distilgpt2 backbone
- 构造简单二分类头
- 用占位样本训练

后续需要补充：
- 真实数据集加载，例如 human/ai 文本 CSV/JSONL
- train/val/test 切分
- 指标评估：accuracy、precision、recall、F1、AUC
- checkpoint metadata：数据集版本、模型版本、训练参数
- 和 detection/text_detector.py 的 load_weights 对齐
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


class TextDataset(Dataset):
    """文本训练数据集。

    texts 是文本列表，labels 中 0 表示 human，1 表示 ai。
    """

    def __init__(self, texts, labels, tokenizer, max_length=512):
        """保存数据和 tokenizer 配置。"""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """返回样本数量。"""
        return len(self.texts)

    def __getitem__(self, idx):
        """把一条文本转换成模型输入。"""
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TextClassifier(nn.Module):
    """基于 transformer backbone 的文本二分类模型。"""

    def __init__(self, model_name="distilgpt2"):
        """加载 backbone，并接一个二分类头。"""
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask):
        """前向传播，输出 human/ai logits。"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, -1, :]
        return self.classifier(pooled)


def train():
    """训练文本二分类模型。

    当前只用占位数据跑通流程。真正训练前需要替换数据加载部分。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # TODO: 替换为真实数据集加载，例如 data/text/train.jsonl。
    texts = ["This is a sample human text.", "AI generated content example."]
    labels = [0, 1]

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = TextDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TextClassifier("distilgpt2").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")

    Path("data/models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "data/models/text_classifier.pt")
    print("Model saved to data/models/text_classifier.pt")


if __name__ == "__main__":
    # 直接运行脚本时启动训练。
    train()
