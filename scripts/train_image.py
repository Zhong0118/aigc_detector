"""Image AI detection model training script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def build_model():
    model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # TODO: Replace with actual dataset paths
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
    train()
