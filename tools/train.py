
from tqdm import tqdm
import math
from functools import partial
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import torch
import random
from torch import Tensor


class EarlyStopping:
    def __init__(self, output, save_interval, patience=5, min_delta=0, verbose=False):
        """
        Args:
            patience (int): 当验证损失不再改善时，等待多少个 epoch 停止训练。
            min_delta (float): 损失需要改善的最小值。默认值为 0。
            verbose (bool): 是否打印提示信息。
            path (str): 保存最佳模型的路径。
            save_interval: 保存模型的间隔
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.output = output 
        self.save_interval = save_interval

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, self.output + "/best.pth")
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model, self.output + "/best.pth")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        
        if (epoch + 1) % self.save_interval == 0:
            torch.save(model, f'{self.output}/epoch_{epoch}.pth')

    def save_checkpoint(self, val_loss, model, model_path):
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model, model_path)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def freeze_parameters(model):
    # 冻结模型的所有参数
    for param in model.parameters():
        param.requires_grad = False


def fit_one_epoch(writer: SummaryWriter, trainer, train_dataLoader, test_dataLoader, epoch, EPOCH, loss_history=None):
  

    total_loss = 0

    print('Start Train')
    trainer.train()
    with tqdm(total=len(train_dataLoader),desc=f'Epoch {epoch + 1}/{EPOCH}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_dataLoader):
            loss = trainer.train_step(*batch)
            total_loss  += loss.item()

            pbar.set_postfix(**{'total_loss'    : total_loss / (iteration + 1), 
                                'lr'            : trainer.get_lr()})
            pbar.update(1)

    writer.add_scalar("loss/train", total_loss / len(train_dataLoader), epoch)


def set_seed(seed):
    # 设置 Python 原生的随机数生成器种子
    random.seed(seed)
    # 设置 NumPy 随机数生成器种子
    np.random.seed(seed)
    # 设置 PyTorch 随机数生成器种子
    torch.manual_seed(seed)
    
    # 如果使用 GPU，也需要确保所有GPU操作的一致性
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
    
    # 设置CuDNN的参数来保证结果的一致性
    torch.backends.cudnn.deterministic = True  # 确保每次卷积的计算结果一致
    torch.backends.cudnn.benchmark = False     # 关闭优化，以保证可重复性

