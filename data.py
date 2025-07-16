import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from utils import encode_text

class KHATTDataset(Dataset):
    def __init__(self, image_dir, text_dir, max_len=64):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.max_len = max_len
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))]
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(self.text_dir, txt_name)

        print(f"🧾 Looking for: {txt_path}")
        print(f"🗂 Exists: {os.path.exists(txt_path)}")

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Text file not found: {txt_path}")

        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        with open(txt_path, "r", encoding="utf-8") as f:
            sentence = f.read().strip()

        text_encoded = torch.tensor(encode_text(sentence, self.max_len))

        return image, text_encoded
