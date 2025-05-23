import torch
import torch.nn as nn
from torchvision.models import resnet50

class SeCoResNet(nn.Module):
    def __init__(self):
        super(SeCoResNet, self).__init__()
        # 加载 torchvision 提供的 resnet50
        backbone = resnet50(pretrained=False)

        # 截掉 avgpool 和 fc（我们只要前面的卷积部分）
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # 到 layer4，输出 (B, 2048, 4, 4)

    def forward(self, x):
        return self.encoder(x)  # 返回特征图，shape: (B, 2048, 4, 4)
