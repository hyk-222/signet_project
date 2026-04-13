import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingHead(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super(EmbeddingHead, self).__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.l2_norm = nn.functional.normalize

    def forward(self, x):
        # 生成嵌入
        embedding = self.fc(x)
        # L2 归一化
        embedding = self.l2_norm(embedding, p=2, dim=1)
        return embedding
