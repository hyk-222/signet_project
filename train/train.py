import torch
import torch.optim as optim
import yaml
import time
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.elastic import ElasticTransform
from models.siamese import SiameseNetwork
from data.dataset import SignetDataset
from data.pair_generator import PairGenerator
from data.batch_sampler import PKSampler
from train.eval import Evaluator
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve

class Trainer:

    def __init__(self, config_path):

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # =============================
        # 超参数
        # =============================
        self.margin = self.config['train']['margin']
        self.epochs = self.config['train']['epochs']
        self.iterations_per_epoch = self.config['train']['iterations_per_epoch']

        # =============================
        # Transform（更稳）
        # =============================
        self.transform = transforms.Compose([
            transforms.Resize((150, 220)),

            ElasticTransform(alpha=5, sigma=2, p=0.2),

            transforms.RandomAffine(
                degrees=2,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02)
            ),

            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # =============================
        # Dataset
        # =============================
        self.train_dataset = SignetDataset(
            root_dir=self.config['data']['root_dir'],
            transform=self.transform,
            split='train'
        )

        self.val_dataset = SignetDataset(
            root_dir=self.config['data']['root_dir'],
            transform=self.transform,
            split='val'
        )

        self.test_dataset = SignetDataset(
            root_dir=self.config['data']['root_dir'],
            transform=self.transform,
            split='test'
        )

        # =============================
        # Loader（关键参数优化）
        # =============================
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=PKSampler(
                self.train_dataset,
                P=8,   # ⭐ 推荐
                K=4
            ),
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            PairGenerator(
                self.val_dataset,
                pairs_per_epoch=1000,
                fixed=True
            ),
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            num_workers=2
        )

        self.test_loader = DataLoader(
            PairGenerator(
                self.test_dataset,
                pairs_per_epoch=1000,
                fixed=True
            ),
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            num_workers=2
        )

        # =============================
        # Model
        # =============================
        self.model = SiameseNetwork().to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['train']['learning_rate'],
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs
        )

    # =====================================================
    # 🚀 Batch Hard Triplet（稳定版）
    # =====================================================
    def semi_hard_triplet_loss(self, embeddings, labels):

        # =========================
        # 距离矩阵
        # =========================
        # 建议在 loss 中加入一个很小的 epsilon
        dist = torch.cdist(embeddings, embeddings) + 1e-8

        N = labels.size(0)
        labels = labels.unsqueeze(1)

        mask_pos = (labels == labels.T)
        mask_neg = (labels != labels.T)

        eye = torch.eye(N, device=embeddings.device).bool()
        mask_pos = mask_pos & (~eye)

        losses = []

        for i in range(N):

            pos_idx = mask_pos[i]
            neg_idx = mask_neg[i]

            pos_dist = dist[i][pos_idx]
            neg_dist = dist[i][neg_idx]

            if len(pos_dist) == 0 or len(neg_dist) == 0:
                continue

            # =========================
            # hardest positive（正常）
            # =========================
            hardest_pos = pos_dist.max()

            # =========================
            # semi-hard negative（核心）
            # =========================
            # 条件：pos < neg < pos + margin
            semi_mask = (neg_dist > hardest_pos) & \
                        (neg_dist < hardest_pos + self.margin)

            semi_neg = neg_dist[semi_mask]

            if len(semi_neg) > 0:
                hardest_neg = semi_neg.min()
            else:
                # fallback：用普通最难负样本
                hardest_neg = neg_dist.min()

            loss = F.relu(
                hardest_pos - hardest_neg + self.margin
            )

            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True).to(embeddings.device)

        loss = torch.stack(losses).mean()

        # =========================
        # 防 collapse（必须保留）
        # =========================
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

                # 标准化距离计算
                emb1 = F.normalize(emb1, dim=1)
                emb2 = F.normalize(emb2, dim=1)
                dist = torch.norm(emb1 - emb2, dim=1)

                all_dist.append(dist.cpu())
                all_label.append(label)

        all_dist = torch.cat(all_dist).numpy()
        all_label = torch.cat(all_label).numpy()

        # 计算 ROC 曲线和 EER
        # 注意：在距离度量中，距离越小越可能是正类，所以分数要取负
        fpr, tpr, thresholds = roc_curve(all_label, -all_dist)
        fnr = 1 - tpr

        # EER 是 FPR 与 FNR 相等的点
        eer = fpr[np.nanargmin(np.absolute((fpr - fnr)))]

        # 找最佳准确率对应的阈值
        best_acc = 0
        for t in np.linspace(0, 2, 50):
            acc = ((all_dist < t) == all_label).mean()
            best_acc = max(best_acc, acc)

        print(
            f"[VAL] EER: {eer:.4f} | Best Acc: {best_acc:.4f} | Dist Gap: {all_dist[all_label == 0].mean() - all_dist[all_label == 1].mean():.3f}")

        return eer  # Early Stopping 现在改为根据 EER 停止（越小越好）
    # =====================================================
    def train(self):
        print("🛠️ 执行数据增强自检...")
        # self.check_transforms(num_images=12)
        best_val = float('inf')
        patience = 5
        counter = 0

        for epoch in range(self.epochs):

            self.model.train()

            total_loss = 0.0
            num_batches = 0

            start_time = time.time()

            print(f"\n🚀 Epoch {epoch + 1}/{self.epochs}")

            pbar = tqdm(self.train_loader, leave=False)

            for b_idx, (imgs, labels) in enumerate(pbar):

                if b_idx >= self.iterations_per_epoch:
                    break

                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # =========================
                # forward
                # =========================
                embeddings = self.model.forward_once(imgs)

                # ⭐ 必须：先归一化再放大
                embeddings = F.normalize(embeddings, dim=1) * 10

                loss = self.semi_hard_triplet_loss(
                    embeddings,
                    labels
                )

                # =========================
                # backward
                # =========================
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 5
                )

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{total_loss / num_batches:.4f}"
                })

            epoch_time = time.time() - start_time

            val_metric = self.validate()
            train_loss = total_loss / num_batches

            print(f"✅ Train Loss: {train_loss:.4f} | {epoch_time:.1f}s")
            print(f"📊 Val Gap: {val_metric:.4f}")

            self.scheduler.step()

            # =============================
            # Early Stopping
            # =============================
            if val_metric < best_val:

                best_val = val_metric
                counter = 0

                torch.save(self.model.state_dict(), 'best_model.pth')
                print("✅ Saved Best Model")

            else:
                counter += 1
                print(f"⚠️ No Improve {counter}/{patience}")

            if counter >= patience:
                print("⛔ Early Stopping")
                break

        # =============================
        # TEST
        # =============================
        print("\nTesting Best Model...")

        self.model.load_state_dict(
            torch.load('best_model.pth')
        )

        evaluator = Evaluator(
            self.model,
            self.test_loader,
            self.device
        )

        evaluator.evaluate()

    def check_transforms(self, num_images=8):
        """可视化增强后的图像，并打印统计信息"""
        self.model.eval()

        # 获取一个 Batch
        try:
            imgs, labels = next(iter(self.train_loader))
        except StopIteration:
            print("❌ Error: Train Loader 为空")
            return

        # 打印基础统计，确认数据范围
        print(f"📊 Batch Mean: {imgs.mean():.4f}, Std: {imgs.std():.4f}")

        # 截取
        imgs = imgs[:min(num_images, imgs.size(0))]

        # 反归一化：将 [-1, 1] 映射回 [0, 1] 以便显示
        # 公式：(x * std) + mean
        view_imgs = imgs * 0.5 + 0.5
        view_imgs = torch.clamp(view_imgs, 0, 1)

        # 制作网格
        grid = vutils.make_grid(view_imgs, nrow=4, padding=4, normalize=False)

        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Augmented Samples (Alpha={self.transform.transforms[1].alpha})")
        plt.axis('off')

        save_path = 'debug_transforms.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # 及时关闭防止内存占用
        print(f"📸 增强预览已更新: {save_path}")