import torch
import torch.nn as nn



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ModularCNN(nn.Module):
    def __init__(self, num_residual_blocks, num_fc_layers, input_channels=3, num_classes=6):
        super(ModularCNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.residual_layers = self._make_residual_layers(num_residual_blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layers = self._make_fc_layers(num_fc_layers, 16 * (2 ** (num_residual_blocks - 1)), num_classes)

    def _make_residual_layers(self, num_blocks):
        layers = []
        in_channels = 16
        for i in range(num_blocks):
            out_channels = in_channels * 2 if i % 2 == 1 else in_channels
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_fc_layers(self, num_layers, input_dim, output_dim):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, input_dim // 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(input_dim // 2))
            layers.append(nn.Dropout(0.5))
            input_dim //= 2
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.residual_layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x