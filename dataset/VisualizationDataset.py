import torch.utils
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import numpy as np
import torch.utils.data  
from tools.preprocess import letterbox
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class VisualizationDataset(Dataset):

    def __init__(self, root_dir, input_shape):
        
        self.data = self.load_data(root_dir)
        self.input_shape = input_shape

    def load_data(self, path):
        res = list()
        for idx, name in enumerate(os.listdir(path)):
            img_dir = os.path.join(path, name)
            
            for img in os.listdir(img_dir):
                res.append((os.path.join(img_dir, img), idx))
        return res

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path)
        img, _ = letterbox(img, self.input_shape)

        return np.array(img) / 255.0, label

    def __len__(self):

        return len(self.data)
    

def dataset_collate(batch):
    

    a_imgs = []
    labels = []
    for img_a, label in batch:
        a_imgs.append(torch.tensor(img_a).permute((2, 0, 1))[None, :].type(torch.FloatTensor))
        labels.append(torch.tensor(label).long())

    return torch.cat(a_imgs, dim=0), torch.tensor(labels)
