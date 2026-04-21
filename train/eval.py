import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix
from sklearn.manifold import TSNE


class Evaluator:
    def __init__(self, model, device, save_dir):
        self.model = model
        self.device = device
        self.save_dir = save_dir

        # ✅ 自动创建目录（防止报错）
        os.makedirs(self.save_dir, exist_ok=True)

    # =====================================================
    # 1. 提取 embedding
    # =====================================================
    def compute_embeddings(self, dataloader):
        self.model.eval()

        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:

                if len(batch) == 2:
                    imgs, lbls = batch
                elif len(batch) == 3:
                    imgs, _, lbls = batch
                else:
                    raise ValueError("Unsupported batch format")

                imgs = imgs.to(self.device)

                emb = self.model.forward_once(imgs)
                emb = F.normalize(emb, dim=1)

                embeddings.append(emb.cpu())
                labels.append(lbls.cpu())

        embeddings = torch.cat(embeddings).numpy()
        labels = torch.cat(labels).numpy()

        return embeddings, labels

    # =====================================================
    # 2. 构建 pair（避免 O(N²) 爆炸）
    # =====================================================
    def build_pairs(self, embeddings, labels, max_pairs=100000):
        N = len(labels)

        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((i, j))

        # ✅ 随机采样
        if len(pairs) > max_pairs:
            idx = np.random.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in idx]

        y_true = []
        y_score = []

        for i, j in pairs:
            same = 1 if labels[i] == labels[j] else 0

            # cosine similarity
            sim = np.dot(embeddings[i], embeddings[j])

            y_true.append(same)
            y_score.append(sim)

        return np.array(y_true), np.array(y_score)

    # =====================================================
    # 3. 计算指标（ROC / EER / ACC）
    # =====================================================
    def evaluate_metrics(self, y_true, y_score):

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fnr = 1 - tpr

        # EER
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]

        # Best ACC
        best_acc = 0
        best_t = thresholds[0]

        for t in thresholds:
            pred = (y_score > t).astype(int)
            acc = accuracy_score(y_true, pred)
            if acc > best_acc:
                best_acc = acc
                best_t = t

        cm = confusion_matrix(y_true, (y_score > best_t).astype(int))

        return {
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "best_acc": float(best_acc),
            "best_threshold": float(best_t),
            "confusion_matrix": cm.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }

    # =====================================================
    # 4. ROC 曲线
    # =====================================================
    def plot_roc(self, fpr, tpr, epoch):
        plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()

        path = os.path.join(self.save_dir, f"roc_epoch_{epoch}.png")
        plt.savefig(path, dpi=300)
        plt.close()

    # =====================================================
    # 5. t-SNE
    # =====================================================
    def visualize_tsne(self, embeddings, labels, epoch):
        print("Running t-SNE...")

        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(embeddings)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=labels,
            cmap='tab10',
            s=5
        )

        plt.colorbar(scatter)
        plt.title(f"t-SNE Epoch {epoch}")

        save_path = os.path.join(self.save_dir, f"tsne_{epoch}.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ t-SNE saved: {save_path}")
        plt.close()

    # =====================================================
    # 6. 距离分布（🔥强烈建议保留）
    # =====================================================
    def plot_distance_distribution(self, y_true, y_score, epoch):
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]

        plt.figure()
        plt.hist(pos, bins=50, alpha=0.5, label="Positive")
        plt.hist(neg, bins=50, alpha=0.5, label="Negative")

        plt.legend()
        plt.title("Distance Distribution")

        path = os.path.join(self.save_dir, f"dist_epoch_{epoch}.png")
        plt.savefig(path, dpi=300)
        plt.close()

    # =====================================================
    # 7. 主入口
    # =====================================================
    def run(self, dataloader, epoch):

        print("🔍 Computing embeddings...")
        embeddings, labels = self.compute_embeddings(dataloader)

        print("🔗 Building pairs...")
        y_true, y_score = self.build_pairs(embeddings, labels)

        print("📊 Evaluating metrics...")
        metrics = self.evaluate_metrics(y_true, y_score)

        # ===== 可视化 =====
        self.plot_roc(metrics["fpr"], metrics["tpr"], epoch)
        self.visualize_tsne(embeddings, labels, epoch)
        self.plot_distance_distribution(y_true, y_score, epoch)

        # ===== 保存指标 =====
        report_path = os.path.join(self.save_dir, f"report_epoch_{epoch}.json")
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"""
================ Evaluation Epoch {epoch} ================
EER: {metrics['eer']:.4f}
Best ACC: {metrics['best_acc']:.4f}
Best Threshold: {metrics['best_threshold']:.4f}
=========================================================
""")

        return metrics