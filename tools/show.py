

import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
import cv2 as cv   
from PIL import ImageDraw, ImageFont, Image
import colorsys

INIT_NUM = 100
_hsv_tuples = [(x / INIT_NUM, 1., 1.) for x in range(INIT_NUM)]
_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), _hsv_tuples))
_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), _colors))

def Tensor2Image(x: Tensor):
    image = x.cpu().numpy()
    image = image.transpose((1, 2, 0))
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image = (image * 255).astype(np.uint8)

    return image


def draw_with_label_conf(image, results: np.ndarray, input_shape = (512, 512), class_names = None, font_path = "'model_data/simhei.ttf"):

    top_label = results[:, -1].astype(np.uint8)
    top_conf = results[:, -2]
    top_boxes = results[:, :4]

    font = ImageFont.truetype(font=font_path, size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
    thickness = max((np.shape(image)[0] + np.shape(image)[1]) // input_shape[0], 1)
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#

    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)] if class_names != None else int(c)
        box             = top_boxes[i]
        score           = top_conf[i]

        top, left, bottom, right = box

        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
        right   = min(image.size[0], np.floor(right).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        # print(label, top, left, bottom, right)
        
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=_colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=_colors[c])
        draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        del draw
    return image
def draw(image: Image.Image, boxes_np: np.ndarray, color=(255, 0, 0), thickness=1) -> Image.Image:
    """
    在 numpy 数组上绘制 xyxy 格式的边界框。

    参数:
    - image: Image
    - boxes: 边界框列表，格式为 [(xmin, ymin, xmax, ymax), ...]。
    - color: 边界框的颜色，格式为 (R, G, B)，默认红色。
    - thickness: 边界框的线条粗细，默认 2。

    返回:
    - Image。
    """
    # 将 numpy 数组转换为 PIL Image 对象
    boxes = boxes_np.astype(np.int32).tolist()
    draw = ImageDraw.Draw(image)

    # 绘制每个边界框
    for box in boxes:
        xmin, ymin, xmax, ymax = box[:4]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=thickness)

    return image

def visualize_anchors(anchors, img_size=(600, 600)):
    """
    可视化锚框在空白图像上
    :param anchors: 锚框数组，每个锚框为 [w, h]
    :param img_size: 图像大小 (width, height)
    """
    # 创建一个空白图像
    image = Image.new('RGB', img_size, 'white')
    draw = ImageDraw.Draw(image)
    
    # 计算图像的中心
    img_width, img_height = img_size
    center_x, center_y = img_width // 2, img_height // 2

    # 绘制锚框
    for anchor in anchors:
        w, h = anchor
        # 计算锚框的左上角和右下角坐标
        x1 = center_x - w // 2
        y1 = center_y - h // 2
        x2 = center_x + w // 2
        y2 = center_y + h // 2
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    
    # 显示图像
    plt.imshow(image)
    plt.axis('off')
    plt.show()
