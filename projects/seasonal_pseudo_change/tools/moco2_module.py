# tools/moco2_module.py
import torch
import torch.nn as nn
import torchvision.models as models

class MoCoV2ResNet50(nn.Module):
    def __init__(self):
        super(MoCoV2ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # 去掉FC层，保留2048-d特征

    def forward(self, x):
        x = self.encoder(x)         # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)   # [B, 2048]
        return x
