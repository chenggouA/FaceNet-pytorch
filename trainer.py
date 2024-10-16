
from tools.baseTrainer import base


class Trainer(base):
    
    def __init__(self, device, model, loss_fn, aux_classifier):
        super().__init__(device, model, loss_fn)
        
        self.aux_classifier = aux_classifier
        
        self.clear_loss()
    
    def clear_loss(self):
        self.dist_loss = 0.0
        if self.aux_classifier:
            self.cls_loss = 0.0
    def forward(self, imgs, *args, **kwargs):
        
        outputs = self.model(imgs)
        losses = self.loss_fn(outputs, *args, **kwargs)

        if not isinstance(losses, list):
            losses = [losses]
        return losses, outputs
    
    def get_result_dict(self, losses):
        result_dict = dict()
        self.dist_loss += losses[0].item()
        result_dict['dist_loss'] = self.dist_loss

        if self.aux_classifier:
            self.cls_loss += losses[1].item()
            result_dict['cls_loss'] = self.cls_loss
        

        return result_dict
 
    
from tools.config import Config
from facenet import FaceNet
import torch 

def get_model(config: Config):
    
    device = config['predict.device'] 
    model_path = config['predict.model_path']

    backbone = config['backbone']

    model = FaceNet(backbone)
    model.load_state_dict(torch.load(model_path)['model'])
    model = model.to(device)

    model.eval()

    return model
        