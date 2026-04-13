import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):

        # embeddings: [2B, D]
        # labels: [2B]

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        loss = 0.0
        triplet_count = 0

        for i in range(len(embeddings)):

            anchor = embeddings[i]
            anchor_label = labels[i]

            # 正样本
            pos_mask = labels == anchor_label
            neg_mask = labels != anchor_label

            pos_dist = dist_matrix[i][pos_mask]
            neg_dist = dist_matrix[i][neg_mask]

            if len(pos_dist) > 1 and len(neg_dist) > 0:

                pos_dist = pos_dist[pos_dist > 0]  # 去掉自己

                hardest_pos = pos_dist.max()
                hardest_neg = neg_dist.min()

                triplet_loss = F.relu(
                    hardest_pos - hardest_neg + self.margin
                )

                loss += triplet_loss
                triplet_count += 1

        if triplet_count == 0:
            return torch.tensor(0.0, requires_grad=True).to(embeddings.device)

        return loss / triplet_count