import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## Basic block과 Residual Block을 나눠서 구현 후 ResNet에 합침



class BasicBlock(nn.Module):
    expansion = 1               ## 무슨 용도인지 모르겠음

    def __init__(self, in_channels, out_channels, stride=1, groups=1, 
                 base_width=64, dilation=1, norm_lyaer=None):
        """
        groups, base_width: ResNext 나 Wide ResNet에서 사용
        """
        super(BasicBlock, self).__init__()

        if norm_lyaer is None:
            norm_lyaer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock은 groups=1, base_width=64만 가능')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1은 BasicBlock에서 불가능')
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_lyaer(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_lyaer(out_channels)

        self.stride = stride
        self.residual = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )

    def forward(self, x):
        out = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += self.residual(out)
        x = self.relu(x)
        return x
    
class Bottleneck(nn.Module):
    expansion = 4               ## 블록 내에서 차원을 증가시키는 3번째 conv layer의 확장계수

    def __init__(self, in_channels, out_channels, stride=1, norm_lyaer=None):
        super(Bottleneck, self).__init__()

        if norm_lyaer is None:
            norm_lyaer = nn.BatchNorm2d
        
        # ResNext 나 WideResNet의 경우 사용
        # width = int(out_channels * (base_width / 64.)) * groups
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=1, bias=False)
        self.bn1 = norm_lyaer(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_lyaer(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, stride=stride, padding=1, bias=False)
        self.bn3 = norm_lyaer(out_channels)
        self.residual = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
    
    def forward(self, x):
        out = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn3(x)

        x += self.residual(out)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fx = nn.Linear(512*block.expansion, num_classes)

        self._init_layer()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride))
            self.in_channels = out_channels * block.expansion   # input_channel 업데이트
        return nn.Sequential(*layers)
    
    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Model:
    def resnet18(self):
        return ResNet(BasicBlock, [2, 2, 2, 2])
    
    def resnet34(self):
        return ResNet(BasicBlock, [3, 4, 6, 3])
    
    def resnet50(self):
        return ResNet(Bottleneck, [3, 4, 6, 3])

    def resnet101(self):
        return ResNet(Bottleneck, [3, 4, 23, 3])
    
    def resnet152(self):
        return ResNet(Bottleneck, [3, 8, 36, 3])