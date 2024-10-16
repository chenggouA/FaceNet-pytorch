from tools.loss import ContrastiveLoss, triplet_loss
from torch import nn, Tensor
import torch

class ComputeLoss(nn.Module):

    def __init__(self, aux_classifier, num_classes, loss_type):
        super().__init__()
        self.num_classes = num_classes
        self.cls_loss = nn.CrossEntropyLoss()
        if loss_type == "TripletLoss":
            self.vec_loss = triplet_loss()
        elif loss_type == "ContrastiveLoss":
            self.vec_loss = ContrastiveLoss()
        self.aux_classifier = aux_classifier
        


    def forward(self, outputs, labels):

        if self.aux_classifier:
            vec_outputs, cls_outputs = outputs
        else:
            vec_outputs = outputs

        device = vec_outputs.device
        
        _vec_loss = self.vec_loss(vec_outputs, labels)
    

        if self.aux_classifier:
            _cls_loss = self.cls_loss(cls_outputs, labels)
            return [_vec_loss, _cls_loss, _cls_loss + _vec_loss]
    
        return _vec_loss 
        # return [_cls_loss, _con_loss, _cls_loss + _con_loss]

        