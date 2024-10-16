
from .baseDataset import baseDataset
import numpy as np
import random
class TripletLossDataset(baseDataset):

    
    def get_data_from_indices(self, indices):
        index = random.choice(indices)  # 随机选择一个索引
        label = self.label[index]
        img_data = self.img_path[index]
        return self.load_image(img_data), label

    def load_triad(self):
        # 随机选择一个 anchor 的索引
        anchor_index = random.randint(0, len(self.label) - 1)

        images = np.zeros((3, 3, *self.input_shape), dtype=np.float32)
        labels = np.zeros(3, dtype=np.uint8)

        # 加载 anchor
        images[0] = self.load_image(self.img_path[anchor_index])
        labels[0] = self.label[anchor_index]

        # 找到正样本的索引
        positive_indices = np.where(self.label == self.label[anchor_index])[0]
        # 从正样本中随机选择一个
        if len(positive_indices) > 1:  # 确保有多个正样本可选
            positive_index = random.choice(positive_indices[positive_indices != anchor_index])
        else:
            positive_index = positive_indices[0]  # 只有一个正样本

        images[1], labels[1] = self.load_image(self.img_path[positive_index]), self.label[positive_index]

        # 找到负样本的索引
        negative_indices = np.where(self.label != self.label[anchor_index])[0]
        # 从负样本中随机选择一个
        negative_index = random.choice(negative_indices)

        images[2], labels[2] = self.load_image(self.img_path[negative_index]), self.label[negative_index]

        return images, labels
    def __getitem__(self, index):
        return self.load_triad()