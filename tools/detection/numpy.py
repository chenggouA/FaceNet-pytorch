import numpy as np

def bbox2loc(anchor: np.ndarray, bbox: np.ndarray):
    '''
        bbox: 真实框  xyxy
        anchor: 锚框  xyxy
        计算真实框和对应的锚框之间的差异
    '''

    anchor_xywh = xyxy2xywh(anchor)
    bbox_xywh = xyxy2xywh(bbox)

    width = anchor_xywh[:, 2]
    height = anchor_xywh[:, 3]

    base_width = bbox_xywh[:, 2]
    base_height = bbox_xywh[:, 3]

    dxdy = (bbox_xywh[:, :2] - anchor_xywh[:, :2]) / anchor_xywh[:, 2:]
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)


    return np.hstack((dxdy, dw[:, None], dh[:, None]))



    
    
def bbox_iou(bbox_a: np.ndarray, bbox_b: np.ndarray):
    '''
        bbox_a, bbox_b shape like (num, xyxy)
    '''

    # 计算交集左上角的坐标
    tl = np.maximum(bbox_a[:, np.newaxis, :2], bbox_b[:, :2])
    
    # 计算交集右下角的坐标
    br = np.minimum(bbox_a[:, np.newaxis, 2:], bbox_b[:, 2:])

    # 计算交集的面积
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)

    # 计算bbox_a的面积

    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)

    # 计算bbox_b的面积
    
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    

    iou = area_i / (area_a[:, np.newaxis] + area_b[np.newaxis, :] - area_i)


    return iou



def loc2bbox(src_bbox: np.ndarray, loc: np.ndarray) -> np.ndarray:
    '''
        src_bbox 先验框
        loc 调整值
        
    '''

    n = loc.shape[0]

    # 获取先验框的宽高
    src_width =  src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    
    # 获取先验框的中心点

    ctr_x = src_bbox[:, 0] + 0.5 * src_width
    ctr_y = src_bbox[:, 1] + 0.5 * src_height

    ctr = np.stack([ctr_x, ctr_y, src_width, src_height], axis=1)

    # 获取偏移量(偏移量相对于先验框的宽高)
    dx = loc[:, 0] * src_width
    dy = loc[:, 1] * src_height


    dxdy = np.stack([dx, dy, np.zeros(n), np.zeros(n)], axis=1)

    ctr += dxdy


    dwdh = np.stack([np.zeros(n), np.zeros(n), loc[:, 2], loc[:, 3]], axis=1)
    dwdh = np.exp(dwdh)
    ctr *= dwdh



    return xywh2xyxy(ctr)

'''
    xyxy -> xywh
'''
def xyxy2xywh(xyxy): 
    
    wh = xyxy[:, 2:] - xyxy[:, :2]
    ctr_xy = xyxy[:, :2] + 0.5 * wh
    
    # axis = 1 拼接  
    return np.hstack((ctr_xy, wh))
    
def xywh2xyxy(xywh):
    
    x = xywh[:, 0]
    y = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]


    x1 = x - 0.5 * w
    x2 = x + 0.5 * w
    y1 = y - 0.5 * h
    y2 = y + 0.5 * h 
    
    return np.stack([x1, y1, x2, y2], axis=1)



def nms(bbox, score: np.ndarray, threshold) -> np.ndarray:
    keep = []

    # 按得分降序排序
    order = score.argsort()[::-1]

    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    # 计算每个边界框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # NMS 算法
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # 计算交集的坐标
        _x1 = np.maximum(x1[i], x1[order[1:]])
        _y1 = np.maximum(y1[i], y1[order[1:]])
        _x2 = np.minimum(x2[i], x2[order[1:]])
        _y2 = np.minimum(y2[i], y2[order[1:]])

        # 计算交集的宽度和高度
        w = np.maximum(0, _x2 - _x1 + 1)
        h = np.maximum(0, _y2 - _y1 + 1)
        
        # 计算交集面积
        intersection = w * h
        
        # 计算 IoU
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # 保留 IoU 小于等于阈值的边界框索引
        ids = np.where(iou <= threshold)[0]
        
        # 更新 order，注意要加 1 以跳过当前的 i
        order = order[ids + 1]

    return keep