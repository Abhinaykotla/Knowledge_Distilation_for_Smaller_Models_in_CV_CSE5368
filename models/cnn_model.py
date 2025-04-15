import torch
import torch.nn as nn
from .blocks import ResidualBlock

class CustomSceneCNN(nn.Module):
    def __init__(self, num_residual_blocks, num_fc_layers, num_classes=6, input_size=(3, 150, 150)):
        super(CustomSceneCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.residual_layers = self._make_residual_layers(num_residual_blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Reduces HxW to 1x1

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            dummy_output = self._forward_features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)

        self.fc_layers = self._make_fc_layers(num_fc_layers, self.flattened_size, num_classes)

    def _make_residual_layers(self, num_blocks):
        layers = []
        in_channels = 16
        out_channels = 16

        for i in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
            if i % 2 == 1:
                out_channels *= 2

        return nn.Sequential(*layers)

    def _make_fc_layers(self, num_layers, in_features, num_classes):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, in_features // 2))
            layers.append(nn.BatchNorm1d(in_features // 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.5))
            in_features //= 2
        layers.append(nn.Linear(in_features, num_classes))
        return nn.Sequential(*layers)

    def _forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.residual_layers(x)
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
