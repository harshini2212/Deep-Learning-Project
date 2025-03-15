import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.bottleneck_channels = out_channels // self.expansion
        
        self.conv1 = nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.bottleneck_channels)
        
        self.conv2 = nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.bottleneck_channels)
        
        self.conv3 = nn.Conv2d(self.bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ModifiedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, initial_channels=64, drop_rate=0.0):
        super(ModifiedResNet, self).__init__()
        self.in_channels = initial_channels
        self.drop_rate = drop_rate

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        
        # Create ResNet layers
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels*8, num_blocks[3], stride=2)
        
        # Final classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(initial_channels*8*block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
            
        out = self.fc(out)
        return out

# Create different ResNet configurations
def resnet18(num_classes=10, initial_channels=64, drop_rate=0.0):
    return ModifiedResNet(BasicBlock, [2, 2, 2, 2], num_classes, initial_channels, drop_rate)

def resnet34(num_classes=10, initial_channels=64, drop_rate=0.0):
    return ModifiedResNet(BasicBlock, [3, 4, 6, 3], num_classes, initial_channels, drop_rate)

def resnet50(num_classes=10, initial_channels=64, drop_rate=0.0):
    return ModifiedResNet(BottleneckBlock, [3, 4, 6, 3], num_classes, initial_channels, drop_rate)

# Compact versions with reduced parameters
def resnet18_compact(num_classes=10, drop_rate=0.0):
    return ModifiedResNet(BasicBlock, [2, 2, 2, 2], num_classes, initial_channels=32, drop_rate=drop_rate)

def resnet34_compact(num_classes=10, drop_rate=0.0):
    return ModifiedResNet(BasicBlock, [3, 4, 6, 3], num_classes, initial_channels=24, drop_rate=drop_rate)