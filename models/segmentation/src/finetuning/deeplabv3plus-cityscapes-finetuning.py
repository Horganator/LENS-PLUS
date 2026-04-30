import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DEEPLAB_PATH = BASE_DIR / "DeepLabV3Plus-Pytorch"

sys.path.insert(0, str(DEEPLAB_PATH))

import network

ROOT = BASE_DIR / "datasets/CITYSCAPES"

NUM_CLASSES = 19
IGNORE_INDEX = 255

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLOR_TO_CLASS = {
    (128, 64, 128): 0,
    (244, 35, 232): 1,
    (70, 70, 70): 2,
    (102, 102, 156): 3,
    (190, 153, 153): 4,
    (153, 153, 153): 5,
    (250, 170, 30): 6,
    (220, 220, 0): 7,
    (107, 142, 35): 8,
    (152, 251, 152): 9,
    (70, 130, 180): 10,
    (220, 20, 60): 11,
    (255, 0, 0): 12,
    (0, 0, 142): 13,
    (0, 0, 70): 14,
    (0, 60, 100): 15,
    (0, 80, 100): 16,
    (0, 0, 230): 17,
    (119, 11, 32): 18,
}

class CityscapesDataset(Dataset):
    def __init__(self, root, split):
        self.img_dir = root / split / "img"
        self.lbl_dir = root / split / "label"

        self.images = sorted(self.img_dir.rglob("*.png"))
        self.labels = sorted(self.lbl_dir.rglob("*.png"))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((360, 640)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.images)

    def encode_rgb_label(self, label):
        h, w, _ = label.shape
        encoded = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
        for color, cls in COLOR_TO_CLASS.items():
            mask = np.all(label == color, axis=-1)
            encoded[mask] = cls
        return encoded

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert("RGB"))
        lbl = np.array(Image.open(self.labels[idx]).convert("RGB"))

        lbl = self.encode_rgb_label(lbl)

        lbl = Image.fromarray(lbl)
        lbl = lbl.resize((640, 360), Image.NEAREST)

        img = self.transform(img)
        lbl = torch.from_numpy(np.array(lbl)).long()

        return img, lbl

def load_model():
    model = network.modeling.__dict__["deeplabv3plus_mobilenet"](
        num_classes=NUM_CLASSES,
        output_stride=8,
    )

    ckpt = torch.load(
        "./deeplabv3plus-mobilenet.pth",
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(ckpt["model_state"])
    return model

def get_loss():
    weights = torch.ones(NUM_CLASSES)
    weights[1] = 2.5
    weights[9] = 3.0

    return nn.CrossEntropyLoss(
        weight=weights.to(DEVICE),
        ignore_index=IGNORE_INDEX
    )

def train():
    train_ds = CityscapesDataset(ROOT, "train")
    val_ds = CityscapesDataset(ROOT, "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = load_model().to(DEVICE)

    criterion = get_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        for imgs, labels in train_bar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        val_loss = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")

        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        print(f"[Epoch {epoch+1}] Val Loss: {val_loss/len(val_loader):.4f}")

        torch.save(model.state_dict(), "./finetuned/deeplab_finetuned.pth")

if __name__ == "__main__":
    train()
