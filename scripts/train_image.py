"""图片 AIGC 检测模型训练脚本。

这个脚本属于后续“本地模型路线”，不是当前 API-first MVP 的必要步骤。
当前内容是训练 scaffold：
- 使用 EfficientNetV2-S ImageNet 预训练权重
- 修改分类头为 human/ai 二分类
- 默认读取 data/images/real 和 data/images/ai

后续需要补充：
- 更丰富的数据集和数据增强
- train/val/test 切分
- 混淆矩阵、AUC、误报率评估
- 保存 checkpoint metadata
- 和 detection/image_detector.py 的 load_weights 对齐
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


class ImageDataset(Dataset):
    """图片训练数据集。

    image_paths 是图片路径列表，labels 中 0 表示 real/human，1 表示 ai。
    """

    def __init__(self, image_paths, labels, transform=None):
        """保存图片路径、标签和 transform。"""
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        """返回样本数量。"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """读取一张图片并转换成 tensor。"""
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def build_model():
    """构建 EfficientNetV2-S 二分类模型。"""
    model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def train():
    """训练图片二分类模型。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # TODO: 替换为真实数据集路径或配置化数据路径。
    data_dir = Path("data/images")
    if not data_dir.exists():
        print("No training data found in data/images/. Please prepare your dataset.")
        print("Expected structure: data/images/real/*.jpg and data/images/ai/*.jpg")
        return

    real_images = list((data_dir / "real").glob("*.*"))
    ai_images = list((data_dir / "ai").glob("*.*"))
    image_paths = real_images + ai_images
    labels = [0] * len(real_images) + [1] * len(ai_images)

    dataset = ImageDataset(image_paths, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0

        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}, Acc: {acc:.4f}")

    Path("data/models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "data/models/image_classifier.pt")
    print("Model saved to data/models/image_classifier.pt")


if __name__ == "__main__":
    # 直接运行脚本时启动训练。
    train()
