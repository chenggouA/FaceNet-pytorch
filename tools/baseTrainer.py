
from torch import nn, Tensor
class base(nn.Module):
    
    def to(self, device) -> None:
        self.model = self.model.to(device)
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __init__(self, device, model, loss_fn):
        super().__init__()
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
    
    def set_optimizer_and_lr_scheduler(self, optimizer, lr_scheduler):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def freeze_backbone(self):
        if hasattr(self.model, "backbone"):
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if hasattr(self.model, "backbone"):
            for param in self.model.backbone.parameters():
                param.requires_grad = True
    
    def train_step(self, *args, **kwargs):


         
        # 清空梯度
        self.optimizer.zero_grad()

        losses, _ = self(*args, **kwargs)

        # 反向传播

        losses[-1].backward()

        self.optimizer.step()

        # 学习率调度器
        self.lr_scheduler.step()
        return losses

    def model_forward(self, input: Tensor):
        return self.model(input)
    def forward(self, input: Tensor, *args, **kwargs):
        
        output = self.model(input.to(self.device))
        
        losses = self.loss_fn(output, *args, **kwargs)

        if not isinstance(losses, list):
            losses = [losses]
        return losses, output
    
    def get_lr(self):
        # 获取每个参数组的组名和学习率
        for param_group in self.optimizer.param_groups:
            lr = param_group.get('lr')
            return lr

