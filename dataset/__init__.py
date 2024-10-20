import torch
from .ContrastLossDataset import ContrastLossDataset
from .LFWDataSet import LFWDataset
from .TripletLossDataset import TripletLossDataset
import numpy as np

def FaceNet_dataset_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    num_image = images.shape[1]
    image_arr = []
    label_arr = []
    for item in range(num_image):
        image_arr.append(images[:, item, ...])
        label_arr.append(labels[:, item, ...])

    images = np.concatenate(image_arr, 0)

    labels = np.concatenate(label_arr, 0)
    
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    labels  = torch.from_numpy(np.array(labels)).long()
    return images, labels

def _FaceNet_dataset_collate(batch):
    # 假设 batch 中每个元素的结构是 ([item_1, item_2, ...], [label_1, label_2, ...])
    num_samples = len(batch)
    
    # 提取第一个样本的形状，以便预分配张量
    items_shapes = [item.shape for item in batch[0][0]]
    labels_shapes = [label.shape for label in batch[0][1]]

    # 预分配张量
    items_tensors = [torch.empty((num_samples, *shape), dtype=torch.float) for shape in items_shapes]
    labels_tensors = [torch.empty((num_samples, *shape), dtype=torch.long) for shape in labels_shapes]

    # 填充数据
    for i, (items, labels) in enumerate(batch):
        for j, item in enumerate(items):
            items_tensors[j][i] = torch.from_numpy(item)
        for k, label in enumerate(labels):
            labels_tensors[k][i] = torch.tensor(label)

    # 将 items 连接在一起
    concatenated_items = torch.cat(items_tensors, dim=0)
    concatenated_labels = torch.cat(labels_tensors, dim=0)
    
    return concatenated_items, concatenated_labels
