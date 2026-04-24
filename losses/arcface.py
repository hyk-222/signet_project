import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        """
        ArcFace 层
        :param in_features: 你的 embedding 维度 (128)
        :param out_features: 你的类别总数 (76)
        :param s: 放大系数 (Scale factor)，小数据集建议 30
        :param m: 角度 Margin (Angular margin)
        """
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # 这就是所有类别的“中心锚点” (Weights)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 1. 归一化输入特征和权重
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # 2. 计算增加了 margin 的 cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        # 放松 margin 约束，防止梯度爆炸
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 3. 构造 one-hot mask
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # 4. 把 margin 加到对应的正确类别上
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 5. 乘以缩放尺度
        output *= self.s

        return output