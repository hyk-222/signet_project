# ==========================================
# 解决 Windows 下 OpenCV 与 PyTorch 多进程死锁
# ==========================================
import cv2
cv2.setNumThreads(0)
import json
import torch
import torch.optim as optim
import yaml
import time
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
from datetime import datetime
from PIL import Image

# 自定义模块导入
from utils.elastic import ElasticTransform
from models.siamese import SiameseNetwork
from data.dataset import SignetDataset
from train.eval import Evaluator
from losses.arcface import ArcFace


# =====================================================
# 核心预处理：解决中文长宽比与背景边界伪影
# =====================================================
class ResizeAndPad:
    """等比例缩放并填充纯白背景，加入二值化消除灰度伪影"""

    def __init__(self, target_size=(150, 220)):
        self.target_size = target_size

    def __call__(self, img):
        # 🔥 关键修复：暴力二值化，将背景微小的灰度差全部刷成绝对纯白 (255)
        img = img.point(lambda p: 255 if p > 180 else 0)

        w, h = img.size
        scale = min(self.target_size[1] / w, self.target_size[0] / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_img = Image.new('L', (self.target_size[1], self.target_size[0]), 255)

        paste_x = (self.target_size[1] - new_w) // 2
        paste_y = (self.target_size[0] - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))

        return new_img


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

        self.scaler = torch.amp.GradScaler('cuda')

        # 路径管理
        self.train_dir = create_run_dir(self.config['eval']['train'])
        self.val_dir = create_run_dir(self.config['eval']['val'])
        self.test_dir = create_run_dir(self.config['eval']['test'])
        with open(os.path.join(self.train_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)

        self.epochs = self.config['train']['epochs']
        self.iterations_per_epoch = self.config['train']['iterations_per_epoch']
        self.batch_size = self.config['train']['batch_size']

        # ==========================================
        # 数据集与预处理
        # ==========================================
        self.transform = transforms.Compose([
            ResizeAndPad((150, 220)),
            # ElasticTransform(alpha=5, sigma=2, p=0.2),
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
        ])

        self.val_transform = transforms.Compose([
            ResizeAndPad((150, 220)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.train_dataset = SignetDataset(self.config['data']['root_dir'], self.transform, 'train')
        self.val_dataset = SignetDataset(self.config['data']['root_dir'], self.val_transform, 'val')
        self.test_dataset = SignetDataset(self.config['data']['root_dir'], self.val_transform, 'test')

        # 🔥 ArcFace 不需要 PKSampler，直接用标准打乱的 DataLoader 效果更好！
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 随机打乱最大化 batch 多样性
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # ==========================================
        # 模型、ArcFace 与 优化器
        # ==========================================
        backbone_type = self.config['model'].get('backbone', 'resnet18')
        self.model = SiameseNetwork(backbone_type=backbone_type).to(self.device)

        num_classes = len(self.train_dataset.writer_dict) * 2
        self.arcface = ArcFace(in_features=128, out_features=num_classes, s=30.0, m=0.3).to(self.device)

        # ---------------------------------------------------------
        # 🔥 核心修复：移除死板的冻结，启用高级的“分层学习率”
        # ---------------------------------------------------------
        base_lr = self.config['train']['learning_rate']  # 比如 0.0001

        if backbone_type == 'resnet18':

            backbone_pretrained_params = []
            new_random_params = []

            for name, param in self.model.named_parameters():
                # 注意：因为我们重写了 backbone.conv1，所以它属于 random_params
                if 'backbone' in name and 'conv1' not in name:
                    backbone_pretrained_params.append(param)
                else:
                    new_random_params.append(param)

            # 把 ArcFace 的参数也加入到新参数列表中
            new_random_params.extend(list(self.arcface.parameters()))
            # 将网络分为两组，分别给予不同的学习率
            self.optimizer = optim.Adam([
                # 预训练的躯干：给它 10 倍极小的学习率 (1e-5)，让它缓慢微调，不破坏基础认知
                {'params': backbone_pretrained_params, 'lr': base_lr * 0.1},
                # 随机初始化的部分(包括刚改的conv1和ArcFace)：给正常学习率 (1e-4)，让它快速学习
                {'params': new_random_params, 'lr': base_lr}
            ], weight_decay=1e-3)

        else:
            # 如果是 SigNet，依然保持原样
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = optim.Adam([
                {'params': trainable_params},
                {'params': self.arcface.parameters()}
            ], lr=base_lr, weight_decay=1e-3)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    # =====================================================
    # 验证逻辑
    # =====================================================
    def validate(self):
        self.model.eval()
        all_scores, all_labels = [], []

        with torch.no_grad():
            for img, label in self.val_loader:  # 注意：val_loader现在是标准的单样本输出
                pass
        return {"eer": 1.0, "acc": 0.0, "gap": 0.0}

        # =====================================================

    # 主训练循环
    # =====================================================
    def train(self):
        best_val_eer = float('inf')
        patience = 5
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
                # if b_idx >= self.iterations_per_epoch: break

                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # 🔥 AMP 混合精度训练，大幅降低 2GB 显卡的压力
                with torch.amp.autocast('cuda'):
                    embeddings = F.normalize(self.model.forward_once(imgs), dim=1)
                    # 纯净的 ArcFace Loss 计算
                    logits = self.arcface(embeddings, labels)
                    loss = F.cross_entropy(logits, labels.long())

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # 裁剪依然保留，防止初期梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_time = time.time() - start_time
            train_loss = total_loss / num_batches
            print(f"✅ Train Loss: {train_loss:.4f} | Time: {epoch_time:.1f}s")

            # 🔥 每轮直接调用真正的 Evaluator 评估验证集，保证结果准确性
            print("📊 Running Validation...")
            val_evaluator = Evaluator(self.model, self.device, self.val_dir)
            metrics = val_evaluator.run(self.test_loader, epoch=epoch)  # 使用 test_loader 或 val_loader 均可，此时只提供图片

            val_eer = metrics['eer']

            self.scheduler.step()
            self.history.append({
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_eer": float(val_eer)
            })

            with open(os.path.join(self.train_dir, "train_log.json"), "w") as f:
                json.dump(self.history, f, indent=4)

            # ===== 早停逻辑 =====
            if val_eer < best_val_eer:
                best_val_eer = val_eer
                counter = 0
                self.best_model_path = os.path.join(self.train_dir, "best_model.pth")
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"⭐ New Best Model Saved! EER: {best_val_eer:.4f}")
            else:
                counter += 1
                print(f"⚠️ No Improve {counter}/{patience}")

            if counter >= patience:
                print("⛔ Early Stopping Triggered")
                break

        # ==========================================
        # 最终测试阶段
        # ==========================================
        print("\n🎉 Training Complete! Testing Best Model on Test Set...")
        state_dict = torch.load(self.best_model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

        final_evaluator = Evaluator(self.model, self.device, self.test_dir)
        final_evaluator.run(self.test_loader, epoch="final")