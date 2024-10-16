
from PIL import Image
import math
import numpy as np
import random
import cv2
import os

class CachedMosaicAugmentor:
    def __init__(self, img_size, cache_dir=None):
        self.img_size = img_size  # 目标图像的尺寸
        self.cache = {}  # 内存缓存
        self.cache_dir = cache_dir  # 磁盘缓存目录
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)  # 如果磁盘缓存目录不存在，创建它

    def mosaic_augment(self, images, labels):
        """对传入的 4 张图像进行 Mosaic 增强"""
        h, w, _ = images[0].shape
        mosaic_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # 左上角
        mosaic_img[:h//2, :w//2] = images[0][:h//2, :w//2]
        # 右上角
        mosaic_img[:h//2, w//2:] = images[1][:h//2, w//2:]
        # 左下角
        mosaic_img[h//2:, :w//2] = images[2][h//2:, :w//2]
        # 右下角
        mosaic_img[h//2:, w//2:] = images[3][h//2:, w//2:]

        # 需要调整标签，以适应拼接后的图像
        mosaic_labels = labels  # 省略细节，此处应该调整标签
        return mosaic_img, mosaic_labels

    def random_sample(self, dataset, idx):
        """从 dataset 中随机选择 3 张额外的图像进行 Mosaic"""
        indices = [idx] + random.sample(range(len(dataset)), 3)
        images = [dataset[i][0] for i in indices]
        labels = [dataset[i][1] for i in indices]
        return images, labels

    def is_cached(self, idx):
        """检查是否有缓存"""
        return idx in self.cache or (self.cache_dir and os.path.exists(self.cache_file_path(idx)))

    def cache_file_path(self, idx):
        """生成磁盘缓存文件路径"""
        return os.path.join(self.cache_dir, f"mosaic_{idx}.npy")

    def load_from_cache(self, idx):
        """从缓存中加载数据"""
        if idx in self.cache:
            # 从内存缓存中加载
            return self.cache[idx]
        elif self.cache_dir:
            # 从磁盘缓存中加载
            file_path = self.cache_file_path(idx)
            if os.path.exists(file_path):
                cached_data = np.load(file_path, allow_pickle=True)
                return cached_data[0], cached_data[1]
        return None

    def save_to_cache(self, idx, mosaic_image, mosaic_labels):
        """保存数据到缓存"""
        # 保存到内存
        self.cache[idx] = (mosaic_image, mosaic_labels)

        # 保存到磁盘
        if self.cache_dir:
            file_path = self.cache_file_path(idx)
            np.save(file_path, [mosaic_image, mosaic_labels])

    def apply_mosaic(self, dataset, idx):
        """执行 Mosaic 增强，检查缓存"""
        if self.is_cached(idx):
            # 从缓存加载增强数据
            return self.load_from_cache(idx)
        
        # 随机选择 3 张其他图像并进行 Mosaic 增强
        images, labels = self.random_sample(dataset, idx)
        mosaic_image, mosaic_labels = self.mosaic_augment(images, labels)

        # 保存到缓存
        self.save_to_cache(idx, mosaic_image, mosaic_labels)
        return mosaic_image, mosaic_labels

def preprocess_input(image):
    
    '''
        image: 单张RGB图像
        都图像进行归一化和标准化
    '''

    image   = np.array(image, dtype = np.float32) / 255.0
    # 假设图像已经缩放到 [0, 1] 范围
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image  - mean) / std

    return image / 255.0

def letterbox_reverse(image: np.ndarray, size):
    nh, nw = size

    H, W = image.shape[0], image.shape[1]
    
    ori_img = image[(H - nh) // 2: (H - nh) // 2 + nh, (W - nw) // 2: (W - nw) // 2 + nw]
    
    return ori_img
def letterbox(image: Image.Image, output_shape = (512, 512)):
    
    w, h = image.size

    H, W = output_shape
    # 长边缩放
    scale = min(W / w, H / h)
    
    nw = math.ceil(w * scale)
    nh = math.ceil(h * scale)

    dx = (W - nw) // 2
    dy = (H - nh) // 2

    image = image.resize((nw, nh))

    new = Image.new("RGB", (W, H), (128, 128, 128))
    new.paste(image, (dx, dy))

    return new, (nh, nw)

def rand(a = 0, b = 1):
    return random.uniform(a, b)


def augment_color_and_aspect(image: Image.Image, box, jitter=.3, hue=.1, sat=0.7, val=0.4):
        '''
            色域变换和宽高扰动
        '''
        #   获得图像的高宽与目标高宽
        iw, ih  = image.size

        #   对图像进行缩放并且进行长和宽的扭曲
        new_ar = iw / ih * rand(1 - jitter,1 + jitter) / rand(1 - jitter,1 + jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * ih)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * iw)
            nh = int(nw / new_ar)
        
        image = image.resize((nw,nh), Image.BICUBIC)

        #   翻转图像
        flip = rand() <.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
       
        #   对图像进行色域变换
        #   计算色域变换的参数
       
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #   将图像转到HSV上
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)) # rgb - hsv
        dtype           = image_data.dtype
        
        #   应用变换
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB) # hsv -> rgb

        #   对真实框进行调整
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]] * nw / iw
            box[:, [1,3]] = box[:, [1,3]] * nh / ih
            if flip: 
                box[:, [0, 2]] = nw - box[:, [2, 0]]
            box[:, 0: 2][box[:, 0: 2] < 0] = 0
            box[:, 2][box[:, 2] > nw] = nw
            box[:, 3][box[:, 3] > nh] = nh
            # box_w = box[:, 2] - box[:, 0]
            # box_h = box[:, 3] - box[:, 1]
            
            # 过滤掉w 或者 h <= 1 的box
            # box = box[np.logical_and(box_w > 1, box_h > 1)] 
        
        image = Image.fromarray(image_data)
        return image, box

def letterbox_and_adjust_bbox(image: Image.Image, box, output_shape = (512, 512)):
    
    w, h = image.size

    H, W = output_shape
    # 长边缩放
    scale = min(W / w, H / h)
    
    nw = math.ceil(w * scale)
    nh = math.ceil(h * scale)

    dx = (W - nw) // 2
    dy = (H - nh) // 2

    image = image.resize((nw, nh))

    new = Image.new("RGB", (W, H), (128, 128, 128))
    new.paste(image, (dx, dy))

    # 调整边界框 
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
        box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
        box[:, 0: 2][box[:, 0: 2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h

    return new, box
