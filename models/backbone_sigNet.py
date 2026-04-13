import torch
import torch.nn as nn
import torch.nn.functional as F


class SigNet(nn.Module):
    """
    原论文 SigNet Backbone
    Input: (1, 155, 220)
    Output: 128-d feature embedding
    """

    def __init__(self):
        super(SigNet, self).__init__()

        # ======================
        # Convolution Block 1
        # ======================
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=0
        )

        self.bn1 = nn.BatchNorm2d(96)

        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2
        )

        # ======================
        # Convolution Block 2
        # ======================
        self.conv2 = nn.Conv2d(
            96,
            256,
            kernel_size=5,
            stride=1,
            padding=2
        )

        self.bn2 = nn.BatchNorm2d(256)

        self.pool2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2
        )

        # ======================
        # Convolution Block 3
        # ======================
        self.conv3 = nn.Conv2d(
            256,
            384,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # ======================
        # Convolution Block 4
        # ======================
        self.conv4 = nn.Conv2d(
            384,
            256,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size=3,
            stride=2
        )

        # ======================
        # 动态推导Flatten维度
        # ======================
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 155, 220)

            x = self._forward_conv(dummy)

            flatten_dim = x.view(1, -1).shape[1]

        print(f"[SigNet] Flatten dim auto-calculated: {flatten_dim}")

        # ======================
        # Fully Connected
        # ======================
        self.fc1 = nn.Linear(flatten_dim, 1024)

        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 128)

    def _forward_conv(self, x):
        """
        卷积部分forward（供初始化动态shape使用）
        """

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.conv3(x))

        x = self.pool3(F.relu(self.conv4(x)))

        return x

    def forward(self, x):

        x = self._forward_conv(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.dropout1(x)

        x = self.fc2(x)

        x = F.normalize(x, p=2, dim=1)

        return x