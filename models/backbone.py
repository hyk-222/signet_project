import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18_Signature(nn.Module):
    """
    ResNet18 for Signature Verification
    Output: 128-d embedding
    """

    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)

        # ======================
        # 1. 改输入通道（1通道）
        # ======================
        self.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # ======================
        # 2. 去掉maxpool（避免信息丢失）
        # ======================
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = nn.Identity()

        # ======================
        # 3. backbone
        # ======================
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # ======================
        # 4. 替换 BN → GN（关键）
        # ======================
        self._replace_bn_with_gn()

        # ======================
        # 5. embedding head
        # ======================
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # 🔥 加一个 BN 稳定分布
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),  # 🔥 新增：随机丢弃 40% 的神经元，防止死记硬背
            nn.Linear(256, embedding_dim)
        )

    def _replace_bn_with_gn(self):
        """
        将所有 BN 替换为 GroupNorm
        """

        def convert(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, nn.GroupNorm(32, child.num_features))
                else:
                    convert(child)

        convert(self)

    def forward(self, x):

        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # GAP
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # embedding
        x = self.embedding(x)

        # L2 normalize（签名任务必须）
        x = F.normalize(x, p=2, dim=1)

        return x