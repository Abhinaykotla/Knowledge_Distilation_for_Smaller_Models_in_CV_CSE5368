import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.utils as vutils
import pandas as pd


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, num_layers):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            layers.append(ConvBlock(channels, base_channels * 2 ** i))
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
            layers.append(ConvBlock(channels, base_channels * 2 ** i))
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
        self.train_losses = []
        self.val_losses = []
        self.teacher = None  # To be set externally after loading teacher

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        img, mask, gt = batch
        pred = self.forward(img)
        loss = F.l1_loss(pred * mask, gt * mask)
        self.train_losses.append(loss.item())

        if batch_idx % 200 == 0:
            out_dir = os.path.join(self.logger.log_dir, "samples", f"train_epoch{self.current_epoch}_batch{batch_idx}")
            os.makedirs(out_dir, exist_ok=True)
            vutils.save_image(pred, os.path.join(out_dir, "pred.png"))
            vutils.save_image(gt, os.path.join(out_dir, "gt.png"))
            vutils.save_image(img[:, :3, :, :], os.path.join(out_dir, "input.png"))

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask, gt = batch
        pred = self.forward(img)
        loss = F.l1_loss(pred * mask, gt * mask)
        self.val_losses.append(loss.item())

        if batch_idx % 200 == 0:
            out_dir = os.path.join(self.logger.log_dir, "samples", f"val_epoch{self.current_epoch}_batch{batch_idx}")
            os.makedirs(out_dir, exist_ok=True)
            vutils.save_image(pred, os.path.join(out_dir, "pred.png"))
            vutils.save_image(gt, os.path.join(out_dir, "gt.png"))
            vutils.save_image(img[:, :3, :, :], os.path.join(out_dir, "input.png"))

        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        log_dir = self.logger.log_dir
        df = pd.DataFrame(self.train_losses, columns=["train_loss"])
        df.to_csv(os.path.join(log_dir, f"train_loss_epoch{self.current_epoch}.csv"), index=False)
        self.train_losses = []

    def on_validation_epoch_end(self):
        log_dir = self.logger.log_dir
        df = pd.DataFrame(self.val_losses, columns=["val_loss"])
        df.to_csv(os.path.join(log_dir, f"val_loss_epoch{self.current_epoch}.csv"), index=False)
        self.val_losses = []

    def set_teacher(self, teacher_model):
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
