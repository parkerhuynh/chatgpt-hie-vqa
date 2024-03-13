import torch
from torch.optim.lr_scheduler import _LRScheduler

class LinearLR(_LRScheduler):
    def __init__(self, optimizer, start_lr, end_lr, num_epochs, last_epoch=-1):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_epochs = num_epochs
        super(LinearLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch > self.num_epochs:
            return [self.end_lr for _ in self.base_lrs]
        lr_decay = (self.start_lr - self.end_lr) / (self.num_epochs - 1)
        lr = self.start_lr - self.last_epoch * lr_decay
        return [lr for _ in self.base_lrs]