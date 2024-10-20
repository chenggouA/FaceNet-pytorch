import torch.utils
from PIL import Image
import torch
from random import random
import numpy as np
import torch.utils.data  
import numpy as np
from PIL import Image
from random import random 
import torch
from .baseDataset import baseDataset

class ContrastLossDataset(baseDataset):

    def load_pair(self, _):

        images = np.zeros((2, 3, *self.input_shape), dtype=np.float32)
        labels = np.zeros(2, dtype=np.uint8)

        # 随机选择第一个图像及其标签
        index1 = np.random.choice(len(self.img_path))
        
        images[0] = self.load_image(self.img_path[index1])
        labels[0] = self.label[index1]

        # 根据 label 选择第二个图像
        if random() <= 0.5:
            # 随机选择一个不同标签的图像
            keep_indices = np.where(self.label != labels[0])[0]
        else:
            # 随机选择一个相同标签的图像（不包括 index1）
            keep_indices = np.where(self.label == labels[0])[0]
            keep_indices = keep_indices[keep_indices != index1]

        index2 = np.random.choice(keep_indices)
        
        labels[1] = self.label[index2]
        images[1] = self.load_image(self.img_path[index2])

        return images, labels

    def __getitem__(self, index):
        return self.load_pair(index)


