import numpy as np

def generate_adaptive_anchors(boxes, k=9, generations=1000, mutation_rate=0.1):
    '''
    生成自适应锚框
    '''

    def iou(box, clusters):
        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])
        intersection = x * y
        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]
        iou_ = intersection / (box_area + cluster_area - intersection + 1e-6)
        return iou_

    def avg_iou(boxes, clusters):
        return np.mean([np.max(iou(box, clusters)) for box in boxes])

    def mutate(anchors, mutation_rate=0.1):
        new_anchors = anchors.copy()
        for i in range(len(anchors)):
            if np.random.rand() < mutation_rate:
                new_anchors[i][0] = new_anchors[i][0] * (1 + np.random.uniform(-0.1, 0.1))
                new_anchors[i][1] = new_anchors[i][1] * (1 + np.random.uniform(-0.1, 0.1))
        return new_anchors
    
    def kmeans(boxes, k, dist=np.median, seed=42):
        np.random.seed(seed)
        clusters = boxes[np.random.choice(boxes.shape[0], k, replace=False)]
        last_clusters = np.zeros((boxes.shape[0],))
        while True:
            distances = 1 - np.array([iou(box, clusters) for box in boxes])
            nearest_clusters = np.argmin(distances, axis=1)
            if (nearest_clusters == last_clusters).all():
                break
            for i in range(k):
                clusters[i] = dist(boxes[nearest_clusters == i], axis=0)
            last_clusters = nearest_clusters
        return clusters, avg_iou(boxes, clusters)

    if k > len(boxes):
        raise ValueError("k 值不能超过 boxes 的数量")
    
    initial_anchors, _ = kmeans(boxes, k)
    best_anchors = initial_anchors

    return best_anchors, None  # 不使用遗传算法(太慢)
    best_fitness = avg_iou(boxes, best_anchors)
    for generation in range(generations):
        new_anchors = mutate(best_anchors, mutation_rate)
        new_fitness = avg_iou(boxes, new_anchors)
        if new_fitness > best_fitness:
            best_anchors = new_anchors
            best_fitness = new_fitness
            print(f"Generation {generation + 1}: Fitness improved to {best_fitness:.4f}")
    return best_anchors, best_fitness


def generate_anchors(scales, ratios, feature_map_size, stride):
    
    '''
        生成锚框
    '''

    anchors = []

    height, width = feature_map_size
    for i in range(height):
        for j in range(width):
            c_x = j * stride
            c_y = i * stride


            for scale in scales:
                for ratio in ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)


                    x1 = c_x - w / 2
                    y1 = c_y - h / 2
                    x2 = c_x + w / 2
                    y2 = c_y + h / 2

                    anchors.append([x1, y1, x2, y2])

    return np.array(anchors)


def generate_anchor_base_by_wh(anchor_wh):
    """
    通过宽高生成锚框的基础矩形
    :param anchor_wh: 锚框的宽高数组, 形状为 (N, 2), 每一行是 [w, h]
    :return: 锚框的基础矩形 [x1, y1, x2, y2]
    """
    anchor_base = np.zeros((len(anchor_wh), 4), dtype=np.float32)
    
    w = anchor_wh[:, 0]
    h = anchor_wh[:, 1]
    
    # 计算锚框的左上角 (x1, y1) 和右下角 (x2, y2) 坐标
    anchor_base[:, 0] = - w / 2  # 左上角 x
    anchor_base[:, 1] = - h / 2  # 左上角 y
    anchor_base[:, 2] = w / 2    # 右下角 x
    anchor_base[:, 3] = h / 2    # 右下角 y
    
    return anchor_base
def generate_anchor_base(base_size=16, 
                         ratios=[0.5, 1, 2], 
                         anchor_scales=[8, 16, 32]):
    """
    生成锚框的基础矩形
    :param base_size: 基础尺寸
    :param ratios: 宽高比
    :param anchor_scales: 锚框的尺度
    :return: 锚框的基础矩形 [x1, y1, x2, y2]
    """
    num_ratios = len(ratios)
    num_scales = len(anchor_scales)
    anchor_base = np.zeros((num_ratios * num_scales, 4), dtype=np.float32)

    for i, ratio in enumerate(ratios):
        for j, scale in enumerate(anchor_scales):
            h = base_size * scale * np.sqrt(ratio)
            w = base_size * scale / np.sqrt(ratio)

            index = i * num_scales + j
            anchor_base[index, 0] = - w / 2.  # 左上角 x
            anchor_base[index, 1] = - h / 2.  # 左上角 y
            anchor_base[index, 2] =   w / 2.  # 右下角 x
            anchor_base[index, 3] =   h / 2.  # 右下角 y
    
    return anchor_base


def _enumerate_shifted_anchor(anchor_base: np.ndarray, feat_stride: int, height: int, width: int):
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift: np.ndarray = np.stack((shift_x.ravel(), shift_y.ravel(),
                     shift_x.ravel(), shift_y.ravel()), axis=1)
    
    
    # 9 个 方框
    A = anchor_base.shape[0] 
    
    # k 个 网格点
    K = shift.shape[0]

    anchor: np.ndarray = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))

    anchor = anchor.reshape((K * A, 4))


    return anchor




def area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

def iou(bbox1, bbox2):
    
    x1 = max(bbox1[0], bbox2[0])
    x2 = min(bbox1[1], bbox2[1])
    y1 = max(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersecion = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    if intersecion < 0: intersecion = 0


    union = area(bbox1) + area(bbox2) - intersecion

    return intersecion * 1.0 / union



