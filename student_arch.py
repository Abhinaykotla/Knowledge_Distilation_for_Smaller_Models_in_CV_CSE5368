import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ConvGnSilu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, num_layers):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            layers.append(ConvGnSilu(channels, base_channels * 2 ** i))
            layers.append(nn.MaxPool2d(kernel_size=2))
            channels = base_channels * 2 ** i
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, base_channels, num_layers):
        super().__init__()
        layers = []
        channels = base_channels * 2 ** (num_layers - 1)
        for i in range(num_layers - 1, -1, -1):
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(ConvGnSilu(channels, base_channels * 2 ** i))
            channels = base_channels * 2 ** i
        layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class StudentInpaintNet(pl.LightningModule):
    def __init__(self, in_channels=4, out_channels=3, base_channels=32, num_layers=3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(in_channels, base_channels, num_layers)
        self.decoder = Decoder(out_channels, base_channels, num_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        img, mask, gt = batch
        pred = self.forward(img)
        loss = F.l1_loss(pred * mask, gt * mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask, gt = batch
        pred = self.forward(img)
        val_loss = F.l1_loss(pred * mask, gt * mask)
        self.log("val_loss", val_loss)
