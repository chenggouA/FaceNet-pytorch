from torch import nn
from models.backbone.inception_resnet_v1 import Inception_ResNet
from tools.train import weights_init
from torch.nn import functional as F
from models.backbone.mobilenet import mobileNet_v2_backbone
import torch
from torch.nn import init

def init_linear(layer):
    # 使用 Xavier 初始化权重
    init.xavier_uniform_(layer.weight)

    # 初始化偏置为常数值（例如 0）
    if hasattr(layer, "bias") and layer.bias != None:
        init.constant_(layer.bias, 0)

class FaceNet(nn.Module):

    def __init__(self, backbone, aux_classifier=False, output_channels=128, num_classes=0):
        super().__init__()
        if backbone == "Inception_resnet_v1":
            self.backbone = Inception_ResNet(output_channels)
            flat_shape = 1792
        elif backbone == "mobilenet_v2":
            self.backbone = mobileNet_v2_backbone("./model_data/mobilenet_v2-b0353104.pth")
            flat_shape = 1280
        else:
            raise ValueError(f"未实现{backbone}")
        
        
        self.linear = nn.Linear(flat_shape, output_channels, bias=False)
        init_linear(self.linear)
        if aux_classifier and num_classes != 0:
            self.classifier = nn.Linear(output_channels, num_classes)
            init_linear(self.classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.last_bn = nn.BatchNorm1d(output_channels, eps=0.001, momentum=0.1, affine=True)
        self.aux_classifier = aux_classifier 

        

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        last = x.view(x.size(0), -1)
        
        last = self.linear(last)
        before_normalize = self.last_bn(last)
        vector = F.normalize(before_normalize, p=2, dim=1) # l2标准化
        if self.aux_classifier:
            cls = self.classifier(before_normalize)
            return vector, cls
        
        return vector


def get_model(backbone, mode, model_path):

    model = FaceNet(backbone, mode)
    model.load_state_dict(torch.load(model_path)['model'], strict=False)

    return model

if __name__ == "__main__":
    model = FaceNet("Inception_resnet_v1", "train")
    import torch

    input = torch.randn((1, 3, 299, 299))
    x = model(input)
    pass
