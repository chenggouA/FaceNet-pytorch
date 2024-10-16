

from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models import vgg16, VGG
from torchvision import models

class VGG_16(nn.Module):
    
    def  __init__(self,
                  blocks) -> None:
        super().__init__()
        self.down_1, self.down_2, self.down_3, self.down_4, self.down_5 = blocks
    
    def forward(self, x: Tensor):
        feat1 = self.down_1(x)
        feat2 = self.down_2(feat1)
        feat3 = self.down_3(feat2)
        feat4 = self.down_4(feat3)
        feat5 = self.down_5(feat4)

        return [feat1, feat2, feat3, feat4, feat5]
def vgg16_backbone(pretrained):

    # 加载预训练的 VGG16 模型
    vgg16 = models.vgg16(pretrained=pretrained)
    
    # 获取 VGG16 的特征部分
    features = vgg16.features
    
    # 保存每个 block
    blocks = []
    current_block = []
    
    for layer in features:
        if isinstance(layer, nn.MaxPool2d):
            # 当遇到 MaxPool2d 时，意味着一个 block 结束
            if len(current_block) != 0:
                blocks.append(nn.Sequential(*current_block))
            current_block = []
        current_block.append(layer)
    return VGG_16(blocks)


if __name__ == "__main__":
    layers1, layers2, layers3, layer4, layer5 = vgg16_backbone(pretrained=True) 

    print(layers1)
    
    
