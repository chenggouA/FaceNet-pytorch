import torch

from torchvision import models

weights = {
    "mobileNet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
}

def mobileNet_v2_backbone(model_path = ""):
    
    model = models.mobilenet_v2()

    if model_path != "":
        model.load_state_dict(torch.load(model_path))
        
    return torch.nn.Sequential(*list(model.children())[:-1])
