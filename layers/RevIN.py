
import torch
import torch.nn as nn


class RevIN(nn.Module):

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):

        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):  # x(32,336,321)
        if mode == 'norm':     
            self._get_statistics(x)
            x = self._normalize(x)    # 把x减去均值然后再除以标准差（0均值，单位标准差）
        elif mode == 'denorm':  # 反归一化
            x = self._denormalize(x)  # (32,96,321) ==>
        else: raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev           # 乘以标准差
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean        # 加上均值
        return x