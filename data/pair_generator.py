import torch
import random
from torch.utils.data import Dataset


class PairGenerator(Dataset):

    def __init__(
            self,
            dataset,
            pairs_per_epoch=10000,
            fixed=False
    ):

        self.dataset = dataset
        self.writer_ids = list(dataset.writer_dict.keys())
        self.pairs_per_epoch = pairs_per_epoch
        self.fixed = fixed

        # 固定验证/测试pair
        if self.fixed:

            self.fixed_pairs = [
                self.generate_pair()
                for _ in range(self.pairs_per_epoch)
            ]

    def __len__(self):
        return self.pairs_per_epoch

    ############################################################
    # 抽离pair生成逻辑
    ############################################################
    def generate_pair(self):

        pair_type = random.random()

        # ==========================
        # Genuine Pair
        # ==========================
        if pair_type < 0.33:

            while True:

                writer_id = random.choice(self.writer_ids)

                genuine_imgs = self.dataset.writer_dict[
                    writer_id
                ]['genuine']

                if len(genuine_imgs) >= 2:
                    break

            img1_path, img2_path = random.sample(
                genuine_imgs,
                2
            )

            label = 1.0

        # ==========================
        # Skilled Forgery
        # ==========================
        elif pair_type < 0.66:

            while True:

                writer_id = random.choice(self.writer_ids)

                genuine_imgs = self.dataset.writer_dict[
                    writer_id
                ]['genuine']

                forgery_imgs = self.dataset.writer_dict[
                    writer_id
                ]['forgery']

                if len(genuine_imgs) > 0 and len(forgery_imgs) > 0:
                    break

            img1_path = random.choice(genuine_imgs)

            img2_path = random.choice(forgery_imgs)

            label = 0.0

        # ==========================
        # Different Writer Pair
        # ==========================
        else:

            while True:

                writer1, writer2 = random.sample(
                    self.writer_ids,
                    2
                )

                genuine1 = self.dataset.writer_dict[
                    writer1
                ]['genuine']

                genuine2 = self.dataset.writer_dict[
                    writer2
                ]['genuine']

                if len(genuine1) > 0 and len(genuine2) > 0:
                    break

            img1_path = random.choice(genuine1)

            img2_path = random.choice(genuine2)

            label = 0.0

        return img1_path, img2_path, label

    ############################################################
    # 获取样本
    ############################################################
    def __getitem__(self, idx):

        if self.fixed:

            img1_path, img2_path, label = self.fixed_pairs[idx]

        else:

            img1_path, img2_path, label = self.generate_pair()

        img1 = self.dataset.load_image(img1_path)
        img2 = self.dataset.load_image(img2_path)

        return (
            img1,
            img2,
            torch.tensor(label, dtype=torch.float32)
        )