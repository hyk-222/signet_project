import torch
import torch.nn as nn
from .backbone_sigNet import SigNet
from .backbone import ResNet18_Signature

class SiameseNetwork(nn.Module):
    # 增加 backbone_type 参数
    def __init__(self, backbone_type='signet'):
        super(SiameseNetwork, self).__init__()

        if backbone_type == 'resnet18':
            print("Loading Pretrained ResNet18 Backbone...")
            # 开启预训练
            self.backbone = ResNet18_Signature(embedding_dim=128, pretrained=True)
        else:
            print("Loading SigNet Backbone...")
            self.backbone = SigNet()

    def forward_once(self, x):
        return self.backbone(x)

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        return emb1, emb2