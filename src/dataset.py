import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PropertyDataset(Dataset):
    def __init__(self, csv_path, image_dir, tabular_cols, target_col=None):
        """
        csv_path      : path to CSV file (train or test)
        image_dir     : directory containing satellite images
        tabular_cols  : list of numeric tabular feature names
        target_col    : price column (for training) or None (for test)
        """

        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.tabular_cols = tabular_cols
        self.target_col = target_col

        for col in tabular_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df[tabular_cols] = self.df[tabular_cols].fillna(0.0)

        if target_col:
            self.df[target_col] = pd.to_numeric(self.df[target_col], errors="coerce")
            self.df[target_col] = self.df[target_col].fillna(0.0)
            self.df[target_col] = np.log1p(self.df[target_col])

        valid_rows = []
        valid_image_indices = []

        for idx in self.df.index:
            img_path = os.path.join(self.image_dir, f"{idx}.png")
            if os.path.exists(img_path):
                valid_rows.append(idx)
                valid_image_indices.append(idx)

        self.df = self.df.loc[valid_rows].reset_index(drop=True)
        self.image_indices = valid_image_indices

        print(f"[INFO] Dataset size after image filtering: {len(self.df)}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        original_idx = self.image_indices[idx]

        img_path = os.path.join(self.image_dir, f"{original_idx}.png")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tabular_values = self.df.loc[idx, self.tabular_cols].astype(float).values
        tabular = torch.tensor(tabular_values, dtype=torch.float32)

        if self.target_col:
            target = torch.tensor(
                float(self.df.loc[idx, self.target_col]),
                dtype=torch.float32
            )
            return image, tabular, target

        return image, tabular
