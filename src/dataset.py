import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class NeuralNavigatorDataset(Dataset):
    def __init__(self, data_dir, max_seq_len=20):
        self.image_dir = os.path.join(data_dir, "images")
        self.ann_dir = os.path.join(data_dir, "annotations")
        self.files = sorted(os.listdir(self.ann_dir))
        self.max_seq_len = max_seq_len

        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def encode_text(self, text):
        vocab = {
            "go":1, "to":2,
            "red":3, "blue":4, "green":5,
            "circle":6, "square":7
        }
        if not isinstance(text, str):
            return torch.zeros(1, dtype=torch.long)
        return torch.tensor(
            [vocab.get(w.lower(),0) for w in text.split()],
            dtype=torch.long
        )

    def pad_path(self, path):
        path = path[:self.max_seq_len]
        while len(path) < self.max_seq_len:
            path.append([0.0,0.0])
        return torch.tensor(path, dtype=torch.float32)

    def _find_image(self, ann):
        # find any .png value
        for v in ann.values():
            if isinstance(v,str) and v.lower().endswith(".png"):
                return v
        raise ValueError("Image filename not found")

    def __getitem__(self, idx):
        with open(os.path.join(self.ann_dir, self.files[idx])) as f:
            ann = json.load(f)

        image_name = ann.get("image_file") or ann.get("image") or self._find_image(ann)
        text = ann.get("text") or ann.get("command") or ""
        path = ann["path"]

        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        image = self.transform(image)

        text = self.encode_text(text)
        path = self.pad_path(path)

        return image, text, path
