from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import torch.utils.data  
from tools.preprocess import letterbox
from dataclasses import dataclass
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

def proprecces(image: np.ndarray):
    # 输入应为 h, w, c
    image = image / 255.0
    # 假设图像已经缩放到 [0, 1] 范围
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image  - mean) / std

    return image.transpose((2, 0, 1))

class baseDataset(Dataset):
    def __len__(self):
        return len(self.label)

    def __init__(self, txt_file_path, input_shape):
        self.img_path = []
        self.label = []
        
        with open(txt_file_path, mode='r', encoding='utf-8') as f:
            for line in f:
                line: str = line.strip()
                img1_path, id = line.split(" ")
                self.img_path.append(img1_path)
                self.label.append(int(id))

        self.label = np.array(self.label)
        self.img_path = np.array(self.img_path)
        self.input_shape = input_shape
        
    def load_image(self, img_path):
        img1 = Image.open(img_path)
        img1, _ = letterbox(img1, self.input_shape)
        return proprecces(np.array(img1, dtype=np.float32))