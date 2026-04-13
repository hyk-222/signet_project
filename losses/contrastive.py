import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.5):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):

        distance = F.pairwise_distance(emb1, emb2)

        loss = torch.mean(
            label * torch.pow(distance, 2) +
            (1 - label) * torch.pow(
                torch.clamp(self.margin - distance, min=0.0),
                2
            )
        )

        return loss