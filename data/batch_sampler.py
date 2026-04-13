import random
from collections import defaultdict
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """
    每个 batch:
    P个writer × K个样本
    """

    def __init__(self, dataset, P=8, K=4):
        self.dataset = dataset
        self.P = P
        self.K = K

        self.label_to_indices = defaultdict(list)

        for idx, label in enumerate(dataset.labels):
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())

    def __iter__(self):
        while True:
            selected_labels = random.sample(self.labels, self.P)
            batch = []
            for label in selected_labels:

                indices = self.label_to_indices[label]

                if len(indices) >= self.K:
                    batch.extend(random.sample(indices, self.K))
                else:
                    batch.extend(
                        random.choices(indices, k=self.K)
                    )

            yield batch

    def __len__(self):
        return len(self.dataset) // (self.P * self.K)