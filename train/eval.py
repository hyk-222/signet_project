import torch
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


class Evaluator:

    def __init__(self, model, dataloader, device):

        self.model = model
        self.dataloader = dataloader
        self.device = device

    def compute_distance(self, emb1, emb2):
        """
        欧氏距离
        """
        return torch.norm(emb1 - emb2, dim=1)

    def evaluate(self):

        self.model.eval()

        all_distances = []
        all_labels = []

        with torch.no_grad():

            for img1, img2, label in self.dataloader:

                img1 = img1.to(self.device)
                img2 = img2.to(self.device)

                emb1, emb2 = self.model(img1, img2)

                dist = self.compute_distance(emb1, emb2)

                all_distances.extend(dist.cpu().numpy())
                all_labels.extend(label.numpy())

        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)

        # ========= 找最佳阈值 =========
        best_acc = 0
        best_thresh = 0

        for thresh in np.arange(0, 2.01, 0.005):

            preds = (all_distances < thresh).astype(int)

            acc = accuracy_score(all_labels, preds)

            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        print("=" * 50)
        print(f"Best Threshold: {best_thresh:.4f}")
        print(f"Accuracy: {best_acc:.4f}")

        # ========= EER =========
        fpr, tpr, thresholds = roc_curve(all_labels, -all_distances)

        fnr = 1 - tpr

        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        print(f"EER: {eer:.4f}")

        return {
            "threshold": best_thresh,
            "accuracy": best_acc,
            "eer": eer
        }