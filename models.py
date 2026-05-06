import torch
import torch.nn as nn


class LightCNN(nn.Module):
    """
    CNN leve inspirada em VGG com ~300K parâmetros.
    Aceita imagens de qualquer número de canais e tamanho espacial.
    """

    def __init__(self, in_channels=1, num_classes=10, img_size=28):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Três MaxPool(2) reduzem a dimensão espacial por 8x
        feat_size = img_size // 8
        flat_features = 128 * feat_size * feat_size

        self.classifier = nn.Sequential(
            nn.Linear(flat_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
