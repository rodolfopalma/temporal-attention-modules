import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


# Based on: https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py
class CyclicalLRScheduler(_LRScheduler):
    def __init__(self, optimizer, min_lr, max_lr, cycle_length, last_epoch=-1):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        cycle = 1 + self.last_epoch // self.cycle_length
        x = np.abs((2 * self.last_epoch) / self.cycle_length - 2 * cycle + 1)
        base_height = (self.max_lr - self.min_lr) * np.maximum(0, (1 - x))
        return [self.min_lr + base_height] * len(self.optimizer.param_groups)
