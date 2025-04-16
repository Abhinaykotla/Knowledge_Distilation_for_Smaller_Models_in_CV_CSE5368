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

class CustomSceneCNN6(nn.Module):
    def __init__(self):
        super(CustomSceneCNN6, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = ResidualBlock(16, 16)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.layer3 = ResidualBlock(32, 32)
        self.layer4 = ResidualBlock(32, 64, stride=2)
        self.layer5 = ResidualBlock(64, 64)
        self.layer6 = ResidualBlock(64, 128, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 64)
        self.fc_bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 6)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    

class CustomSceneCNN8(nn.Module):
    def __init__(self):
        super(CustomSceneCNN8, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = ResidualBlock(16, 16)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.layer3 = ResidualBlock(32, 32)
        self.layer4 = ResidualBlock(32, 64, stride=2)
        self.layer5 = ResidualBlock(64, 64)
        self.layer6 = ResidualBlock(64, 128, stride=2)
        self.layer7 = ResidualBlock(128, 128)
        self.layer8 = ResidualBlock(128, 256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.fc_bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 6)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    

class CustomSceneCNN10(nn.Module):
    def __init__(self):
        super(CustomSceneCNN10, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = ResidualBlock(16, 16)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.layer3 = ResidualBlock(32, 32)
        self.layer4 = ResidualBlock(32, 64, stride=2)
        self.layer5 = ResidualBlock(64, 64)
        self.layer6 = ResidualBlock(64, 128, stride=2)
        self.layer7 = ResidualBlock(128, 128)
        self.layer8 = ResidualBlock(128, 256, stride=2)
        self.layer9 = ResidualBlock(256, 256)
        self.layer10 = ResidualBlock(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 6)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x