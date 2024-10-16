from .backbone import vgg16_backbone, resnet50_backbone

from .centernet import CenterNet
def centerNet_vgg16(pretrained, num_classes, model_path = None):
    
    return CenterNet(vgg16_backbone(pretrained, model_path), in_channel=512, num_classes=num_classes)

def centerNet_resnet50(pretrained, num_classes, mode_path = None):
    
    return CenterNet(resnet50_backbone(pretrained, mode_path), in_channel=2048, num_classes=num_classes) 
