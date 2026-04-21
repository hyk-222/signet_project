import json
import torch
import torch.optim as optim
import yaml
import time
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from losses.contrastive import ContrastiveLoss
from losses.triplet import TripletLoss
from utils.elastic import ElasticTransform
from models.siamese import SiameseNetwork
from data.dataset import SignetDataset
from data.pair_generator import PairGenerator
from data.batch_sampler import PKSampler
from train.eval import Evaluator
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve
import os
from datetime import datetime


def create_run_dir(base_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

class Trainer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ===== 路径管理=====
        self.train_dir = create_run_dir(self.config['eval']['train'])
        self.val_dir = create_run_dir(self.config['eval']['val'])
        self.test_dir = create_run_dir(self.config['eval']['test'])
        # 保存配置（可复现实验）
        with open(os.path.join(self.train_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)
        # 超参数
        self.margin = self.config['train']['margin']
        self.epochs = self.config['train']['epochs']
        self.iterations_per_epoch = self.config['train']['iterations_per_epoch']
        # ===== 动态 margin =====
        self.initial_margin = self.margin
        self.final_margin = self.margin * 0.5
        # Loss 函数选择
        self.loss_name = self.config['train'].get('loss_type', 'hybrid_triplet_loss')
        if self.loss_name == 'triplet':
            self.criterion = TripletLoss(margin=self.margin)
        elif self.loss_name == 'contrastive':
            self.criterion = ContrastiveLoss(margin=1.5)

        # ===== transform =====
        self.transform = transforms.Compose([
            transforms.Resize((150, 220)),
            ElasticTransform(alpha=5, sigma=2, p=0.2),
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((150, 220)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # ===== dataset =====
        self.train_dataset = SignetDataset(
            self.config['data']['root_dir'],
            self.transform,
            'train'
        )

        self.val_dataset = SignetDataset(
            self.config['data']['root_dir'],
            self.val_transform,
            'val'
        )

        self.test_dataset = SignetDataset(
            self.config['data']['root_dir'],
            self.val_transform,
            'test'
        )

        # ===== loader =====
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=PKSampler(self.train_dataset, P=8, K=4),
            num_workers=4, pin_memory=True
        )

        self.val_loader = DataLoader(
            PairGenerator(self.val_dataset, pairs_per_epoch=1000, fixed=True),
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            num_workers=4
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            num_workers=4
        )
        # 模型与优化器
        self.model = SiameseNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['train']['learning_rate'],
                                    weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    def compute_loss(self, embeddings, labels):
        """统一 Loss 计算接口，分流至不同算法"""
        # if self.loss_name == 'semi_hard_triplet':
        #     return self.semi_hard_triplet_loss(embeddings, labels)
        if self.loss_name == 'batch_hard_triplet':
            return self.batch_hard_triplet_loss(embeddings, labels)
        elif self.loss_name == 'hybrid_triplet':
            return self.hybrid_triplet_loss(embeddings, labels)
        elif self.loss_name == 'triplet':
            return self.criterion(embeddings, labels)
        elif self.loss_name == 'contrastive':
            # 将 PKSampler 产出的 Batch 转换为对偶样本
            emb1, emb2 = embeddings[0::2], embeddings[1::2]
            # 若 label 相同则为正对 (1)，不同为负对 (0)
            pair_labels = (labels[0::2] == labels[1::2]).float()
            return self.criterion(emb1, emb2, pair_labels)

    def semi_hard_triplet_loss(self, embeddings, labels):
        """原有的 Semi-Hard 逻辑，包含正则项防止崩塌"""
        dist = torch.cdist(embeddings, embeddings) + 1e-8
        N = labels.size(0)
        labels_sq = labels.unsqueeze(1)
        mask_pos = (labels_sq == labels_sq.T) & (~torch.eye(N, device=self.device).bool())
        mask_neg = (labels_sq != labels_sq.T)

        losses = []
        for i in range(N):
            pos_dist, neg_dist = dist[i][mask_pos[i]], dist[i][mask_neg[i]]
            if len(pos_dist) == 0 or len(neg_dist) == 0: continue

            hardest_pos = pos_dist.max()
            semi_mask = (neg_dist > hardest_pos) & (neg_dist < hardest_pos + self.margin)
            hardest_neg = neg_dist[semi_mask].min() if semi_mask.any() else neg_dist.min()

            losses.append(F.relu(hardest_pos - hardest_neg + self.margin))

        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True).to(self.device)
        reg = (embeddings.norm(dim=1) - 1).abs().mean()
        return loss + 0.001 * reg

    def batch_hard_triplet_loss(self, embeddings, labels):
        """
        Batch-Hard Triplet Loss
        embeddings: (N, D)
        labels: (N,)
        """

        # pairwise 距离矩阵
        dist = torch.cdist(embeddings, embeddings) + 1e-8

        N = labels.size(0)
        labels = labels.unsqueeze(1)

        # mask
        mask_pos = (labels == labels.T)
        mask_neg = (labels != labels.T)

        # 去掉自己
        eye = torch.eye(N, device=embeddings.device).bool()
        mask_pos = mask_pos & ~eye

        # hardest positive（每行最大）
        dist_ap = torch.max(dist * mask_pos.float(), dim=1)[0]

        # hardest negative（每行最小）
        # trick：把非负样本设为 +inf
        dist_an = dist.clone()
        dist_an[~mask_neg] = float('inf')
        dist_an = torch.min(dist_an, dim=1)[0]

        # Triplet Loss
        loss = F.relu(dist_ap - dist_an + self.margin)

        # 去掉无效项（有些可能没有 positive）
        valid = (dist_ap > 0) & (dist_an < float('inf'))
        if valid.any():
            loss = loss[valid].mean()
        else:
            loss = torch.zeros(1, device=embeddings.device, requires_grad=True)

        return loss

    def hybrid_triplet_loss(self, embeddings, labels):
        dist = torch.cdist(embeddings, embeddings) + 1e-8

        N = labels.size(0)
        labels = labels.unsqueeze(1)

        mask_pos = (labels == labels.T)
        mask_neg = (labels != labels.T)

        eye = torch.eye(N, device=embeddings.device).bool()
        mask_pos = mask_pos & ~eye

        losses = []

        for i in range(N):
            pos_dist = dist[i][mask_pos[i]]
            neg_dist = dist[i][mask_neg[i]]

            if len(pos_dist) == 0 or len(neg_dist) == 0:
                continue

            # ===== hardest positive =====
            hardest_pos = pos_dist.max()

            # ===== semi-hard negative（优先）=====
            semi_mask = (neg_dist > hardest_pos) & (neg_dist < hardest_pos + self.margin)

            if semi_mask.any():
                semi_neg = neg_dist[semi_mask].min()
            else:
                # 🔥 限制 hardest negative（防极端值）
                sorted_neg = torch.sort(neg_dist)[0]
                k = min(5, len(sorted_neg))  # 只取前5个最难
                semi_neg = sorted_neg[:k].mean()

            # ===== soft margin =====
            loss = F.softplus(hardest_pos - semi_neg + self.margin)

            losses.append(loss)

        if len(losses) == 0:
            return torch.zeros(1, device=embeddings.device, requires_grad=True)

        loss = torch.stack(losses).mean()

        # ===== embedding 正则（🔥非常重要）=====
        reg = (embeddings.norm(dim=1) - 1).abs().mean()

        return loss + 0.001 * reg
    # =====================================================
    # 验证（用距离）
    # =====================================================

    def validate(self):
        self.model.eval()

        all_dist = []
        all_label = []

        with torch.no_grad():
            for img1, img2, label in self.val_loader:
                img1, img2 = img1.to(self.device), img2.to(self.device)

                emb1, emb2 = self.model(img1, img2)
                emb1 = F.normalize(emb1, dim=1)
                emb2 = F.normalize(emb2, dim=1)

                dist = torch.norm(emb1 - emb2, dim=1)

                all_dist.append(dist.cpu())
                all_label.append(label)

        all_dist = torch.cat(all_dist).numpy()
        all_label = torch.cat(all_label).numpy()

        # ===== EER =====
        fpr, tpr, _ = roc_curve(all_label, -all_dist)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fpr - fnr))]

        # ===== ACC =====
        best_acc = 0
        for t in np.linspace(0, 2, 50):
            acc = ((all_dist < t) == all_label).mean()
            best_acc = max(best_acc, acc)

        # ===== GAP =====
        pos = all_dist[all_label == 1]
        neg = all_dist[all_label == 0]
        gap = neg.mean() - pos.mean()

        return {
            "eer": eer,
            "acc": best_acc,
            "gap": gap
        }
    # =====================================================
    def train(self):
        best_val = float('inf')
        patience = 3
        counter = 0

        self.history = []
        for epoch in range(self.epochs):
            self.model.train()

            total_loss = 0.0
            num_batches = 0

            start_time = time.time()

            print(f"\n🚀 Epoch {epoch + 1}/{self.epochs}")

            pbar = tqdm(self.train_loader, leave=False)
            for b_idx, (imgs, labels) in enumerate(pbar):
                progress = epoch / self.epochs
                self.margin = self.initial_margin * (1 - progress) + self.final_margin * progress
                print(f"📉 Current Margin: {self.margin:.4f}")  # 可选（建议加）
                if b_idx >= self.iterations_per_epoch: break

                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                embeddings = F.normalize(self.model.forward_once(imgs), dim=1)

                # 修改点：应用统一的 compute_loss 接口
                loss = self.compute_loss(embeddings, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{total_loss / num_batches:.4f}"
                })

            epoch_time = time.time() - start_time

            val_res = self.validate()
            train_loss = total_loss / num_batches

            print(f"✅ Train Loss: {train_loss:.4f} | {epoch_time:.1f}s")
            print(f"📊 VAL | EER: {val_res['eer']:.4f} | ACC: {val_res['acc']:.4f} | GAP: {val_res['gap']:.4f}")

            self.scheduler.step()
            self.history.append({
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_eer": float(val_res['eer']),
                "val_acc": float(val_res['acc']),
                "val_gap": float(val_res['gap'])
            })
            with open(os.path.join(self.train_dir, "train_log.json"), "w") as f:
                json.dump(self.history, f, indent=4)
            # Early Stopping
            # =============================
            val_metric = val_res['eer']
            if val_metric < best_val:

                best_val = val_metric
                counter = 0
                best_model_path = os.path.join(self.train_dir, "best_model.pth")
                self.best_model_path = best_model_path
                torch.save(self.model.state_dict(), best_model_path)
                print("✅ Saved Best Model")

            else:
                counter += 1
                print(f"⚠️ No Improve {counter}/{patience}")

            if counter >= patience:
                print("⛔ Early Stopping")
                break
        print("\nTesting Best Model...")

        state_dict = torch.load(
            self.best_model_path,
            map_location=self.device,
            weights_only=True  # ✅ 修复 warning
        )
        self.model.load_state_dict(state_dict)

        # ===== TEST =====
        evaluator = Evaluator(self.model, self.device, self.test_dir)
        metrics = evaluator.run(self.test_loader, epoch="final")