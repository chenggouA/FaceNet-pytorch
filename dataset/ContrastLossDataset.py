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
    def load_image(self, img_path):
        img1 = Image.open(img_path)
        img1
    def load_pair(self, index):
        item_1 = self.load_image(self.img_path[index])
        label_1 = self.label[index]

        if random() <= 0.5:
            keep_indices = np.where(self.label != label_1)[0]
        else:
            keep_indices = np.where(self.label == label_1)[0]
            keep_indices = keep_indices[keep_indices != index]

        index2 = np.random.choice(keep_indices)
        label_2 = self.label[index2]
        item_2 = self.load_image(self.img_path[index2])

        return item_1, item_2, label_1, label_2

    def __getitem__(self, index):
        return self.load_pair(index)


