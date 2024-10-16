import torch
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
from .baseDataset import proprecces



class LFWDataset(Dataset):
    def __init__(self, dir, pairs_path, image_size, transform=None):
        super().__init__()
        self.image_size = image_size
        self.pairs_path = pairs_path
        self.validation_images = self.get_lfw_paths(dir)

    def read_lfw_pairs(self,pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return pairs

    def get_lfw_paths(self,lfw_dir,file_ext="jpg"):

        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        for i in range(len(pairs)):
        #for pair in pairs:
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                path_list.append((path0,path1,issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    def __getitem__(self, index):
        (path_1, path_2, issame)    = self.validation_images[index]
        image1, image2              = Image.open(path_1), Image.open(path_2)

        img1, _ = letterbox(image1, self.image_size)
        img2, _ = letterbox(image2, self.image_size)
        
        img1 = proprecces(np.array(img1, dtype=np.float32))
        img2 = proprecces(np.array(img2, dtype=np.float32))
        
        return torch.from_numpy(img1).type(torch.FloatTensor), torch.from_numpy(img2).type(torch.FloatTensor), issame


    def __len__(self):
        return len(self.validation_images)


