
import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import numpy as np 
import cv2
import random
from tools.preprocess import augment_color_and_aspect, letterbox_and_adjust_bbox    

class DetectionDataset(Dataset):

    def __init__(self, cache_dir, input_shape):
        self.cache_dir = cache_dir
        self.input_shape = input_shape
        os.makedirs(cache_dir, exist_ok=True)  # 创建缓存目录
    
    def mosaic(self, a, b, c, d):

        def resize(img, bboxes, h, w):
            ih, iw = img.shape[0: -1]
            scalar = max(h / ih, w / iw)

            
            img = cv2.resize(img, (0, 0), fx=scalar, fy=scalar)
            bboxes[..., :-1] = bboxes[..., :-1] * scalar
            
            return img, bboxes
        
        def adjustment_bboxes(bboxes, xyxy, offset):
            # 确保 bboxes 是 NumPy 数组
            bboxes = np.array(bboxes)
            
            # 提取 xyxy 中的坐标
            x1, y1, x2, y2 = xyxy
            
            bboxes[:, 0] += offset[1]
            bboxes[:, 1] += offset[0]
            bboxes[:, 2] += offset[1]
            bboxes[:, 3] += offset[0]

            bboxes_w = bboxes[:, 2] - bboxes[:, 0]
            bboxes_h = bboxes[:, 3] - bboxes[:, 1]

            # 对每个 bbox 进行裁剪，确保它们不会超出 Mosaic 区域的边界
            bboxes[:, 0] = np.clip(bboxes[:, 0], x1, x2)  # xmin
            bboxes[:, 1] = np.clip(bboxes[:, 1], y1, y2)  # ymin
            bboxes[:, 2] = np.clip(bboxes[:, 2], x1, x2)  # xmax
            bboxes[:, 3] = np.clip(bboxes[:, 3], y1, y2)  # ymax
            
            # 过滤宽高大于1的框
            bboxes = bboxes[np.logical_and(bboxes_w > 2, bboxes_h > 2)]
        
            return bboxes
        
        img_a, bboxes_a = a
        img_b, bboxes_b = b
        img_c, bboxes_c = c
        img_d, bboxes_d = d 

        a = np.array(img_a)
        b = np.array(img_b)
        c = np.array(img_c)
        d = np.array(img_d)

        mosaic = np.full((self.input_shape[0] * 2, self.input_shape[1] * 2, 3), 114, dtype=np.uint8)
        
        yc, xc = [int(random.uniform(x // 2, x // 2 * 3)) for x in self.input_shape]
        H, W = [x * 2 for x in self.input_shape]

        # 左上角
        a, bboxes_a = resize(a, bboxes_a, yc, xc)

        # draw(Image.fromarray(a), bboxes_a).show()
        
        h0, w0 = a.shape[0: -1]
        w = int(min(xc, w0))
        h = int(min(yc, h0))

        mosaic[yc - h: yc, xc - w: xc] = a[h0 - h: h0, w0 - w: w0]
        # 调整bbox 
        bboxes_a = adjustment_bboxes(bboxes_a, (xc - w, yc - h, xc, yc), (yc - h0, xc - w0))

        # 右上角
        b, bboxes_b = resize(b, bboxes_b, yc, W - xc)
        h1, w1 = b.shape[0: -1]
        w = int(min(W - xc, w1))
        h = int(min(yc, h1))

        mosaic[yc - h: yc, xc: xc + w] = b[h1 - h: h1, 0: w]
        bboxes_b = adjustment_bboxes(bboxes_b, (xc, yc - h, xc + w, yc), (yc - h1, xc))

        # 左下角
        c, bboxes_c = resize(c, bboxes_c, H - yc, xc)
        h2, w2 = c.shape[0: -1]
        w = int(min(xc, w2))
        h = int(min(H - yc, h2))

        mosaic[yc: yc + h, xc - w: xc] = c[0: h, w2 - w: w2]
        bboxes_c = adjustment_bboxes(bboxes_c, (xc - w, yc, xc, yc + h), (yc, xc - w2))

        # 右下角
        d, bboxes_d = resize(d, bboxes_d, H - yc, W - xc)
        h3, w3 = d.shape[0: -1]
        w = int(min(W - xc, w3))
        h = int(min(H - yc, h3))

        mosaic[yc: yc + h, xc: xc + w] = d[0: h, 0: w]
        bboxes_d = adjustment_bboxes(bboxes_d, (xc, yc, xc + w, yc + w), (yc, xc))
        
        
        bboxes = np.vstack([bboxes_a, bboxes_b, bboxes_c, bboxes_d])
        
        image = cv2.resize(mosaic, (0, 0), fx=0.5, fy=0.5)
        bboxes[..., :-1] = bboxes[..., :-1] * 0.5

        return image, bboxes

    def save_mosaic(self, mosaic_image, bboxes, cache_path):
        # 保存 Mosaic 图像及其对应的边界框信息
        cv2.imwrite(f'{cache_path}.jpg', mosaic_image)
        np.save(f'{cache_path}_bboxes.npy', bboxes)

    def load_mosaic(self, cache_path):
        # 从硬盘加载 Mosaic 图像及其对应的边界框
        mosaic_image = cv2.imread(f'{cache_path}.jpg')
        bboxes = np.load(f'{cache_path}_bboxes.npy')
        return mosaic_image, bboxes

    def mosaic_exists(self, cache_path):
        # 检查 Mosaic 文件是否存在
        return os.path.exists(f'{cache_path}.jpg') and os.path.exists(f'{cache_path}_bboxes.npy')


class VOCDataset(DetectionDataset):

    voc_id_to_class = {
        0: "aeroplane",
        1: "bicycle",
        2: "bird",
        3: "boat",
        4: "bottle",
        5: "bus",
        6: "car",
        7: "cat",
        8: "chair",
        9: "cow",
        10: "dog",
        11: "horse",
        12: "motorbike",
        13: "person",
        14: "pottedplant",
        15: "sheep",
        16: "sofa",
        17: "train",
        18: "tvmonitor",
        19: "diningtable"
    }

    voc_class_to_id = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "dog": 10,
        "horse": 11,
        "motorbike": 12,
        "person": 13,
        "pottedplant": 14,
        "sheep": 15,
        "sofa": 16,
        "train": 17,
        "tvmonitor": 18,
        "diningtable": 19
    }


    
    def __init__(self, root, year, cache_dir, input_shape, image_set='train'):
        super().__init__(cache_dir=cache_dir, input_shape=input_shape)

        self.root = root
        self.year = year
        self.image_set = image_set

        voc_root = os.path.join(self.root, f'VOC{self.year}')
        image_set_file = os.path.join(voc_root, 'ImageSets', 'Main', 
                                      f'{self.image_set}.txt')

        with open(image_set_file) as f:
            self.file_names = f.read().splitlines()
        
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(voc_root, 'Annotations')

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = {}
            for dc in map(self.parse_voc_xml, children):
                for k, v in dc.items():
                    if k in def_dic:
                        if isinstance(def_dic[k], list):
                            def_dic[k].append(v)
                        else:
                            def_dic[k] = [def_dic[k], v]
                    else:
                        def_dic[k] = v
            voc_dict[node.tag] = def_dic
        else:
            voc_dict[node.tag] = node.text
        return voc_dict

    def __len__(self):
        return len(self.file_names)
    
    def extract_bbox(self, obj):
        """
        提取对象中的边界框坐标
        """
        bndbox = obj['bndbox']
        bbox = [int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])]
        return bbox
    
    def load_detection_data(self, idx):

        image, targets = self.load_data(idx)

        bboxes = targets['bboxes'] # xyxy

        # 数据增强
        if self.image_set == "train":
            image, bboxes = augment_color_and_aspect(image, bboxes)
        
        # image, bboxes = letterbox_and_adjust_bbox(image, bboxes)

        
        bboxes = np.hstack((bboxes, np.array(targets['labels'])[:, None]))
        
        

        return image, bboxes
    
    def load_data(self, idx):
        file_name = self.file_names[idx]

        # 加载图像
        img_path = os.path.join(self.image_dir, f'{file_name}.jpg')
        img = Image.open(img_path).convert("RGB")

        # 加载标注 XML 文件
        annotation_path = os.path.join(self.annotation_dir, f'{file_name}.xml')
        
        annotation = self.parse_voc_xml(ET.parse(annotation_path).getroot())['annotation']

        # 提取边界框和标签
        bboxes = []
        labels = []
        
        # 检查是否有对象(object)
        if 'object' in annotation:
            objects = annotation['object']
            # 如果有多个对象
            if isinstance(objects, list):
                for obj in objects:
                    bboxes.append(self.extract_bbox(obj))
                    labels.append(VOCDataset.voc_class_to_id[obj['name']])
            # 如果只有一个对象
            else:
                bboxes.append(self.extract_bbox(objects))
                labels.append(VOCDataset.voc_class_to_id[objects['name']])

        # 构建返回的字典
        target = {
            'bboxes': np.array(bboxes), 
            'labels': np.array(labels)   
        }

        return img, target
    
    def __getitem__(self, idx):
        
        if self.image_set != "train":
            
            images, bboxes = self.load_detection_data(idx)

            return letterbox_and_adjust_bbox(images, bboxes, self.input_shape)
        # 组合 4 张图片的索引，其中 idx 是第一张图片
        img_indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        
        # 只使用第一张图片的 ID 作为缓存的关键索引
        cache_name = f"mosaic_{idx}"
        cache_path = os.path.join(self.cache_dir, cache_name)

        if self.mosaic_exists(cache_path):
            # 如果缓存存在，直接加载
            return self.load_mosaic(cache_path)
        else:
            # 加载原始图像数据并生成 Mosaic
            data = [self.load_detection_data(i) for i in img_indices]
            mosaic_image, bboxes = self.mosaic(*data)
            
            # 保存生成的 Mosaic 图像和边界框到硬盘
            self.save_mosaic(mosaic_image, bboxes, cache_path)
            return mosaic_image, bboxes        
 


