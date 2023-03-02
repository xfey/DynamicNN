import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Identity', 'FactorizedReduce', 'Zero']


class BasicLayer(nn.Module):
    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
    
    def partialChannel(self, mode, value):
        pass


class ConvBnRelu(BasicLayer):
    def __init__(self, C_in, C_out, kernel_size, stride, padding=1):
        super().__init__(C_in, C_out, stride)

        self.op = nn.Sequential(
            nn.Conv2d(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(C_out, affine=True, momentum=0.997, eps=1e-5),
            nn.ReLU(inplace=False),
        )
    
    def partialChannel(self, mode, value):
        ...

    def forward(self, x):
        return self.op(x)



class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class FactorizedReduce(BasicLayer):
    """
    Reduce feature map size by factorized pointwise (stride=2).
    """

    def __init__(self, C_in, C_out):
        super().__init__(C_in, C_out, stride=2)
        assert C_out % 2 == 0, "C_out should be even!"
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=True)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.0
        # re-sizing by stride
        return x[:, :, :: self.stride, :: self.stride] * 0.0
