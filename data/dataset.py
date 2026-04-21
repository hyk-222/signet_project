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
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
    ):

        self.root_dir = root_dir
        self.transform = transform

        self.writer_dict = {}

        self.samples = []   # [(img_path, label)]
        self.labels = []    # [label1, label2, ...]

        # 解析 + 划分
        self.parse_dataset()
        self.split_dataset(
            split,
            train_ratio,
            val_ratio,
            seed
        )

        self.build_samples()

    # ==================================================
    # 解析数据
    # ==================================================
    def parse_dataset(self):

        org_dir = os.path.join(
            self.root_dir,
            'full_org'
        )

        forg_dir = os.path.join(
            self.root_dir,
            'full_forg'
        )

        # ========= Genuine =========
        for fname in os.listdir(org_dir):

            nums = re.findall(r'\d+', fname)

            if len(nums) == 0:
                print("非法文件：", fname)
                continue

            writer_id = int(nums[0])

            self.writer_dict.setdefault(
                writer_id,
                {
                    'genuine': [],
                    'forgery': []
                }
            )

            self.writer_dict[writer_id]['genuine'].append(
                os.path.join(org_dir, fname)
            )

        # ========= Forgery =========
        for fname in os.listdir(forg_dir):

            nums = re.findall(r'\d+', fname)

            if len(nums) == 0:
                print("非法文件：", fname)
                continue

            writer_id = int(nums[0])

            self.writer_dict.setdefault(
                writer_id,
                {
                    'genuine': [],
                    'forgery': []
                }
            )

            self.writer_dict[writer_id]['forgery'].append(
                os.path.join(forg_dir, fname)
            )

    # ==================================================
    # 划分数据集（按 writer）
    # ==================================================
    def split_dataset(
            self,
            split,
            train_ratio,
            val_ratio,
            seed
    ):

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
            raise ValueError("split must be train/val/test")

        self.writer_dict = {
            k: self.writer_dict[k]
            for k in selected
        }

        print(
            f"{split.upper()} SET: "
            f"{len(self.writer_dict)} writers"
        )

    # ==================================================
    #  samples（Triplet关键）
    # ==================================================
    def build_samples(self):

        self.samples = []
        self.labels = []
        writer_id_map = {
            writer_id: idx
            for idx, writer_id in enumerate(self.writer_dict.keys())
        }

        for writer_id, data in self.writer_dict.items():
            base_label = writer_id_map[writer_id] * 2  # 真实签名 label: 偶数
            forg_label = base_label + 1  # 伪造签名 label: 奇数

            for img_path in data['genuine']:
                self.samples.append((img_path, base_label))
                self.labels.append(base_label)

            for img_path in data['forgery']:
                self.samples.append((img_path, forg_label))
                self.labels.append(forg_label)

        print(f"Total samples: {len(self.samples)}")

    # ==================================================
    # 读取图像
    # ==================================================
    def load_image(self, path):

        img = Image.open(path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img

    # ==================================================
    # PyTorch接口
    # ==================================================
    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        img = self.load_image(img_path)

        return img, label

    def __len__(self):
        return len(self.samples)