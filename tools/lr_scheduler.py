
import math

import math

class YOLOXCosineLR:
    def __init__(self, optimizer, base_lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_iters = total_iters
        self.warmup_iters = max(1, int(warmup_iters_ratio * total_iters))
        self.warmup_lr_start = max(warmup_lr_ratio * base_lr, 1e-6)
        self.current_iter = 0

        # 设置初始学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

    def get_lr(self):
        if self.current_iter < self.warmup_iters:
            lr = (self.base_lr - self.warmup_lr_start) * (self.current_iter / self.warmup_iters) + self.warmup_lr_start
        elif self.current_iter >= self.total_iters:
            lr = self.min_lr
        else:
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * (self.current_iter - self.warmup_iters) / (self.total_iters - self.warmup_iters))
            )
        return lr

    def step(self):
        # 更新当前迭代
        lr = self.get_lr()
        self.current_iter += 1

        # 更新优化器中的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def state_dict(self):
        return {
            'current_iter': self.current_iter,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr,
            'total_iters': self.total_iters,
            'warmup_iters': self.warmup_iters,
            'warmup_lr_start': self.warmup_lr_start,
        }

    def load_state_dict(self, state):
        self.current_iter = state['current_iter']
        self.base_lr = state['base_lr']
        self.min_lr = state['min_lr']
        self.total_iters = state['total_iters']
        self.warmup_iters = state['warmup_iters']
        self.warmup_lr_start = state['warmup_lr_start']
