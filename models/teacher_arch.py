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

        # Updated to include 12 residual blocks
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
        self.layer11 = ResidualBlock(512, 512)
        self.layer12 = ResidualBlock(512, 1024, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Adjusted FC layers to handle increased feature complexity
        self.fc1 = nn.Linear(1024, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 6)

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
        x = self.layer11(x)
        x = self.layer12(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.fc_bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x