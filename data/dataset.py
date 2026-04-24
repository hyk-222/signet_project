import os
import re
import random
from PIL import Image


class SignetDataset:

    def __init__(
            self,
            root_dir,
            transform=None,
            split='train',
            dataset_type='chisig',  # 新增：用于区分数据集格式
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
    ):
        """
        :param dataset_type: 'cedar' (分目录存储) 或 'chisig' (单目录，ID区分真伪)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_type = dataset_type.lower()

        self.writer_dict = {}

        self.samples = []  # [(img_path, label)]
        self.labels = []  # [label1, label2, ...]

        # 1. 执行数据解析
        self.parse_dataset()

        # 2. 划分数据集（按 writer 划分，确保开集验证）
        self.split_dataset(
            split,
            train_ratio,
            val_ratio,
            seed
        )

        # 3. 构建最终样本列表
        self.build_samples()

    # ==================================================
    # 数据解析逻辑
    # ==================================================
    def parse_dataset(self):
        if self.dataset_type == 'cedar':
            # --------- 原有 CEDAR 逻辑：依赖子目录 full_org 和 full_forg ---------
            org_dir = os.path.join(self.root_dir, 'full_org')
            forg_dir = os.path.join(self.root_dir, 'full_forg')

            # 处理真迹
            if os.path.exists(org_dir):
                for fname in os.listdir(org_dir):
                    nums = re.findall(r'\d+', fname)
                    if not nums: continue
                    writer_id = int(nums[0])
                    self.writer_dict.setdefault(writer_id, {'genuine': [], 'forgery': []})
                    self.writer_dict[writer_id]['genuine'].append(os.path.join(org_dir, fname))

            # 处理伪造
            if os.path.exists(forg_dir):
                for fname in os.listdir(forg_dir):
                    nums = re.findall(r'\d+', fname)
                    if not nums: continue
                    writer_id = int(nums[0])
                    self.writer_dict.setdefault(writer_id, {'genuine': [], 'forgery': []})
                    self.writer_dict[writer_id]['forgery'].append(os.path.join(forg_dir, fname))

        elif self.dataset_type == 'chisig':
            # --------- 新增 ChiSig 逻辑：单目录平铺，靠 ID 范围区分真伪 ---------
            # 格式: [姓名]-[ID]-[编号].jpg
            # 逻辑: ID <= 100 为真(writer_id = ID), ID > 100 为伪造(writer_id = ID - 100)
            if not os.path.exists(self.root_dir):
                print(f"❌ 目录不存在: {self.root_dir}")
                return

            for fname in os.listdir(self.root_dir):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                # 使用 '-' 分隔文件名
                parts = fname.split('-')
                if len(parts) < 3:
                    continue

                try:
                    # 提取中间的 ID
                    raw_id = int(parts[1])

                    if raw_id <= 100:
                        # 真实签名
                        writer_id = raw_id
                        category = 'genuine'
                    else:
                        # 伪造签名（针对 raw_id - 100 的那个人）
                        writer_id = raw_id - 100
                        category = 'forgery'

                    self.writer_dict.setdefault(writer_id, {'genuine': [], 'forgery': []})
                    self.writer_dict[writer_id][category].append(os.path.join(self.root_dir, fname))
                except (ValueError, IndexError):
                    continue
        else:
            raise ValueError(f"尚未支持的数据集类型: {self.dataset_type}")

        print(f"成功从 {self.dataset_type} 加载了 {len(self.writer_dict)} 名作者的数据。")

    # ==================================================
    # 划分数据集（按 writer）
    # ==================================================
    def split_dataset(self, split, train_ratio, val_ratio, seed):
        writer_ids = list(self.writer_dict.keys())
        random.seed(seed)
        random.shuffle(writer_ids)

        total = len(writer_ids)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        if split == 'train':
            selected = writer_ids[:train_end]
        elif split == 'val':
            selected = writer_ids[train_end:val_end]
        elif split == 'test':
            selected = writer_ids[val_end:]
        else:
            raise ValueError("split 必须为 train/val/test")

        self.writer_dict = {k: self.writer_dict[k] for k in selected}
        print(f"{split.upper()} 集合: 包含 {len(self.writer_dict)} 名作者")

    # ==================================================
    # 构建样本列表（用于 PKSampler 或 DataLoader）
    # ==================================================
    def build_samples(self):
        self.samples = []
        self.labels = []

        # 将原始 ID 映射为从 0 开始的连续 label
        writer_id_map = {writer_id: idx for idx, writer_id in enumerate(self.writer_dict.keys())}

        for writer_id, data in self.writer_dict.items():
            # 为每个作者分配两个类别：真迹(偶数) 和 伪造(奇数)
            base_label = writer_id_map[writer_id] * 2
            forg_label = base_label + 1

            for img_path in data['genuine']:
                self.samples.append((img_path, base_label))
                self.labels.append(base_label)

            for img_path in data['forgery']:
                self.samples.append((img_path, forg_label))
                self.labels.append(forg_label)

        print(f"总样本数: {len(self.samples)}")

    def load_image(self, path):
        img = Image.open(path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = self.load_image(img_path)
        return img, label

    def __len__(self):
        return len(self.samples)