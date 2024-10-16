

import torch
from torch import Tensor


def loc2bbox(src_bbox: Tensor, loc: Tensor) -> Tensor:
    '''
        src_bbox: 先验框
        loc: 调整值
        根据loc调整先验框
    '''
    device = src_bbox.device  # 获取 src_bbox 的设备
    n = loc.shape[0]

    # 获取先验框的宽高
    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]

    # 获取先验框的中心点
    ctr_x = src_bbox[:, 0] + 0.5 * src_width
    ctr_y = src_bbox[:, 1] + 0.5 * src_height

    ctr = torch.stack([ctr_x, ctr_y, src_width, src_height], axis=1).to(device)

    # 获取偏移量(偏移量相对于先验框的宽高)
    dx = loc[:, 0] * src_width
    dy = loc[:, 1] * src_height

    dxdy = torch.stack([dx, dy, torch.zeros(n, device=device), torch.zeros(n, device=device)], axis=1)

    ctr += dxdy

    dwdh = torch.stack([torch.zeros(n, device=device), torch.zeros(n, device=device), loc[:, 2], loc[:, 3]], axis=1)

    dwdh = torch.exp(dwdh)
    ctr *= dwdh

    return xywh2xyxy(ctr, device)



def roi_pooling(input, rois, output_size):
    """
    Args:
        input (torch.Tensor): 形状为 (batch_size, channels, height, width) 的输入特征图。
        rois (torch.Tensor): 形状为 (num_rois, 5)，其中 5 为 [batch_index, x_min, y_min, x_max, y_max]。
        output_size (tuple): 输出的固定大小 (output_height, output_width)。
    
    Returns:
        torch.Tensor: 形状为 (num_rois, channels, output_height, output_width) 的池化输出。
    """
    output_height, output_width = output_size
    num_rois = rois.size(0)
    channels = input.size(1)

    output = torch.zeros((num_rois, channels, output_height, output_width), device=input.device)

    for i in range(num_rois):
        roi = rois[i]
        batch_index = int(roi[0].item())
        x_min = int(roi[1].item())
        y_min = int(roi[2].item())
        x_max = int(roi[3].item())
        y_max = int(roi[4].item())

        roi_feature_map = input[batch_index, :, y_min:y_max, x_min:x_max]
        pooled_feature_map = F.adaptive_max_pool2d(roi_feature_map, (output_height, output_width))
        
        output[i] = pooled_feature_map

    return output

def xywh2xyxy(xywh, device) -> Tensor:
    
    x = xywh[:, 0]
    y = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]


    x1 = x - 0.5 * w
    x2 = x + 0.5 * w
    y1 = y - 0.5 * h
    y2 = y + 0.5 * h 
    
    return torch.stack([x1, y1, x2, y2], axis=1).to(device)

def nms(bbox, score: Tensor, threshold) -> Tensor:

    device = score.device
    keep = []

    # 按得分降序排序
    order = score.argsort(descending=True)

    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    # 计算每个边界框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # NMS 算法
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        # 计算交集的坐标
        _x1 = torch.maximum(x1[i], x1[order[1:]])
        _y1 = torch.maximum(y1[i], y1[order[1:]])
        _x2 = torch.minimum(x2[i], x2[order[1:]])
        _y2 = torch.minimum(y2[i], y2[order[1:]])

        # 计算交集的宽度和高度
        w = torch.maximum(torch.tensor(0.0, device=device), _x2 - _x1 + 1)
        h = torch.maximum(torch.tensor(0.0, device=device), _y2 - _y1 + 1)
        
        # 计算交集面积
        intersection = w * h
        
        # 计算 IoU
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # 保留 IoU 小于等于阈值的边界框索引
        ids = torch.where(iou <= threshold)[0]
        
        # 更新 order，注意要加 1 以跳过当前的 i
        order = order[ids + 1]

    return torch.tensor(keep, device=device)

from torch.nn import functional as F
from torch import nn
def pool_nms(heatmap, k = 3):
    '''
        热力图nms
    '''
    padding = (k - 1) // 2
    hmax = nn.MaxPool2d(kernel_size=k, padding=padding, stride=1)(heatmap)
    
    keep = (hmax == heatmap).float()
    
    return heatmap * keep