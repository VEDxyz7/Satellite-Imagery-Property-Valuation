import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-2]
        )  # (B, 512, 7, 7)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 512)

        for param in self.features.parameters():
            param.requires_grad = False

        self.feature_maps = None

    def forward(self, x, enable_gradcam=False):
        x = self.features(x)

        if enable_gradcam:
            x.requires_grad_(True)
            x.retain_grad()
            self.feature_maps = x

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class TabularEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)  # (B, 32)

class MultimodalRegressor(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.tabular_encoder = TabularEncoder(tabular_dim)

        self.regressor = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, tabular, enable_gradcam=False):
        img_feat = self.image_encoder(image, enable_gradcam=enable_gradcam)
        tab_feat = self.tabular_encoder(tabular)

        fused = torch.cat([img_feat, tab_feat], dim=1)
        out = self.regressor(fused)

        return out.squeeze(1)
