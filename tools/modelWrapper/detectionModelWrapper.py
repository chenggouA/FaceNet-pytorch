
from torch import nn, Tensor
from PIL import Image
from tools.preprocess import preprocess_input, letterbox
import numpy as np
import torch
from tools.show import draw_with_label_conf
class ModelWrapper(nn.Module):

    def __init__(self, num_classes, model, mode, input_size, device):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.mode = mode
        self.device = device
        self.input_size = input_size
        self.model = self.model.to(device)

        self.model = self.model.eval()

    @property
    def is_cuda(self):
        return self.device == "cuda"
    def preprocess(self, image_pil: Image.Image) -> Tensor:
        image_array = preprocess_input(image_pil)
        # 添加维度
        batch_input = np.expand_dims(np.transpose(image_array, (2, 0, 1)), 0)  
        
        image_tensor = torch.from_numpy(batch_input).to(self.device)
        
        return image_tensor.float()
    
    def postprocess(self, output):
        raise NotImplementedError()
    def forward(self, input):
        with torch.no_grad():
            if self.mode == "image":
                
                image_pil = Image.open(input).convert("RGB")
                # 原图
                image_ori = image_pil.copy()
                
                # letter_box 
                image_pil, size = letterbox(image_pil, self.input_size)

                output = self.model(self.preprocess(image_pil))

                bboxes_six = self.postprocess(output)[0]

                if len(bboxes_six) == 0:
                    return image_pil
                return draw_with_label_conf(image_pil, bboxes_six) 
                
            elif self.mode == 'video':
                pass
            else:
                raise ValueError("模式错误")

    