import torch
from torch.utils.data import DataLoader
import random

class PairGenerator:
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = [dataset[i][1] for i in range(len(dataset))]
        self.label_to_indices = {}
        
        # 构建标签到索引的映射
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
    
    def generate_pairs(self):
        while True:
            anchor_indices = []
            positive_indices = []
            negative_indices = []
            
            for _ in range(self.batch_size):
                # 随机选择一个标签作为锚点
                anchor_label = random.choice(list(self.label_to_indices.keys()))
                
                # 确保该标签至少有2个样本
                if len(self.label_to_indices[anchor_label]) < 2:
                    continue
                
                # 选择锚点和正样本
                anchor_idx, positive_idx = random.sample(self.label_to_indices[anchor_label], 2)
                
                # 选择负样本（不同标签）
                negative_label = random.choice([l for l in self.label_to_indices.keys() if l != anchor_label])
                negative_idx = random.choice(self.label_to_indices[negative_label])
                
                anchor_indices.append(anchor_idx)
                positive_indices.append(positive_idx)
                negative_indices.append(negative_idx)
            
            # 生成批次数据
            anchor_images = torch.stack([self.dataset[i][0] for i in anchor_indices])
            positive_images = torch.stack([self.dataset[i][0] for i in positive_indices])
            negative_images = torch.stack([self.dataset[i][0] for i in negative_indices])
            
            yield anchor_images, positive_images, negative_images

class TripletGenerator:
    def __init__(self, dataset, batch_size=32):
        self.pair_generator = PairGenerator(dataset, batch_size)
    
    def generate_triplets(self):
        return self.pair_generator.generate_pairs()
