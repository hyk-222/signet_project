import torch
import torch.nn as nn
from .backbone_sigNet import SigNet


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.backbone = SigNet()

    def forward_once(self, x):

        return self.backbone(x)

    def forward(self, x1, x2):

        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)

        return emb1, emb2