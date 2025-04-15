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
    


class CustomSceneCNN(nn.Module):
    def __init__(self):
        super(CustomSceneCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Adding 32 layers of ResidualBlock
        self.layer1 = self._make_layer(16, 16, num_blocks=4, stride=1)
        self.layer2 = self._make_layer(16, 32, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(32, 64, num_blocks=4, stride=2)
        self.layer4 = self._make_layer(64, 128, num_blocks=4, stride=2)
        self.layer5 = self._make_layer(128, 256, num_blocks=4, stride=2)
        self.layer6 = self._make_layer(256, 512, num_blocks=4, stride=2)
        self.layer7 = self._make_layer(512, 1024, num_blocks=4, stride=2)
        self.layer8 = self._make_layer(1024, 1024, num_blocks=4, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers updated for deeper architecture
        self.fc1 = nn.Linear(1024, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 6)

        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

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
        x = self.relu(self.fc_bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x