
from torch import Tensor
from tools.baseTrainer import base
class Trainer(base):

    def __init__(self, device, model, optimizer, loss_fn, num_classses):
        super().__init__(device, model, optimizer, loss_fn)

        self.num_classes = num_classses 

    def forward(self, imgs: Tensor, pngs: Tensor, labels: Tensor):

        imgs = imgs.to(self.device)
        pngs = pngs.to(self.device)
        labels = labels.to(self.device)
        
        outputs = self.model(imgs) 

        loss = self.loss_fn(outputs, pngs) 
        return loss, outputs

    def train_step(self, imgs: Tensor, pngs, labels):
        
        # 清空梯度
        self.optimizer.zero_grad()

        losses, _ = self(imgs, pngs, labels)

        # 反向传播

        losses.backward()

        self.optimizer.step()

        return losses
