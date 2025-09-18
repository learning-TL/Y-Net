import torch
import torch.nn as nn
from memory_profiler import profile

from Channel_mamba_model import CM_Branch
from MS_MOE import MS_MOE_SC
from Spatial_mamba_model import SM_Branch
from collections import OrderedDict

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)
def norm(norm_type, nc, dim=2):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True) if not dim == 3 else nn.BatchNorm3d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False) if not dim == 3 else nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer
def conv_block(in_nc, out_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, norm_type=None):
    '''
    Conv layer with padding, normalization, activation
    mode: Conv -> Norm -> Act  (in_nc, out_nc,3,3)
    '''
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
                  dilation=dilation, bias=bias, groups=groups)
    a = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    n = norm(norm_type, out_nc)
    return sequential(c, n, a)

class DenseBlock_5C(nn.Module):
    '''
    Dense Block
    style: 5 convs
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, padding=1, bias=True, norm_type='batch'):
        super(DenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, padding=padding, bias=bias, norm_type=norm_type)
        self.conv2 = conv_block(nc + gc, gc, kernel_size, stride, padding=padding, bias=bias, norm_type=norm_type)
        self.conv3 = conv_block(nc + 2 * gc, gc, kernel_size, stride, padding=padding, bias=bias, norm_type=norm_type)
        self.conv4 = conv_block(nc + 3 * gc, gc, kernel_size, stride, padding=padding, bias=bias, norm_type=norm_type)
        self.conv5 = conv_block(nc + 4 * gc, nc, kernel_size, stride, padding=padding, bias=bias, norm_type=norm_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x

class Y_MBNet1(nn.Module):
    def __init__(self, in_channels, him_channels, out_channels):
        super(Y_MBNet1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, him_channels, kernel_size=3, stride=1, padding=1),  # 2倍下采样卷积层
            nn.BatchNorm2d(him_channels),
            nn.ReLU())
        self.SM_model1 = SM_Branch(hidden_dim=him_channels)
        self.SM_model2 = SM_Branch(hidden_dim=2 * him_channels + 2)
        self.SM_model3 = SM_Branch(hidden_dim=4 * him_channels + 2)
        self.sdown_conv1 = nn.Sequential(
            nn.Conv2d(him_channels, 2 * him_channels, kernel_size=3, stride=2, padding=1),  # 2倍下采样卷积层
            nn.BatchNorm2d(2 * him_channels),
            nn.ReLU(),
        )
        self.sdown_conv2 = nn.Sequential(
            nn.Conv2d(2 * him_channels + 2, 4 * him_channels, kernel_size=3, stride=2, padding=1),  # 2倍下采样卷积层
            nn.BatchNorm2d(4 * him_channels),
            nn.ReLU(),
        )
        self.CM_model1 = CM_Branch(hidden_dim=him_channels)
        self.CM_model2 = CM_Branch(hidden_dim=2 * him_channels + 2)
        self.CM_model3 = CM_Branch(hidden_dim=4 * him_channels + 2)
        self.cdown_conv1 = nn.Sequential(
            nn.Conv2d(him_channels, 2 * him_channels, kernel_size=3, stride=2, padding=1),  # 2倍下采样卷积层
            nn.BatchNorm2d(2 * him_channels),
            nn.ReLU(),
        )
        self.cdown_conv2 = nn.Sequential(
            nn.Conv2d(2 * him_channels + 2, 4 * him_channels, kernel_size=3, stride=2, padding=1),  # 4倍下采样卷积层
            nn.BatchNorm2d(4 * him_channels),
            nn.ReLU(),
        )
        self.msmoe = MS_MOE_SC(num_channels=him_channels)
        # 2倍上采样1
        self.upsample_2x1 = nn.Sequential(
            nn.Conv2d(9 * him_channels, 9 * him_channels, kernel_size=3, stride=1, padding=1),# 卷积层
            DenseBlock_5C(9 * him_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(9 * him_channels, 9 * him_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        # 2倍上采样2
        self.upsample_2x2 = nn.Sequential(
            nn.Conv2d(9 * him_channels, 6 * him_channels, kernel_size=3, stride=1, padding=1),  # 卷积层
            DenseBlock_5C(6 * him_channels),  # 卷积层
            nn.ReLU(),
            nn.ConvTranspose2d(6 * him_channels, 6 * him_channels, kernel_size=3, stride=2, padding=1, output_padding=0)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(6 * him_channels, 6 * him_channels, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.BatchNorm2d(6 * him_channels),
            nn.ReLU(),
            nn.Conv2d(6 * him_channels, out_channels, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.ReLU(),
        )

    # @profile
    def forward(self, input):
        input = self.conv1(input)
        # 输入图像2，4下采用
        input2x = input[:, :, ::2, ::2]
        input4x = input[:, :, ::4, ::4]
        # print(f"x图像：B={input.shape[0]}, C={input.shape[1]}, H={input.shape[2]}, W={input.shape[3]}")
        # print(f"2x图像：B={input2x.shape[0]}, C={input2x.shape[1]}, H={input2x.shape[2]}, W={input2x.shape[3]}")
        # print(f"4x图像：B={input4x.shape[0]}, C={input4x.shape[1]}, H={input4x.shape[2]}, W={input4x.shape[3]}")
        # 空间 MB
        x1c = self.SM_model1(input)
        x = self.sdown_conv1(x1c)
        x2c = self.SM_model2(torch.cat((x, input2x), dim=1))
        x = self.sdown_conv2(x2c)
        x3c = self.SM_model3(torch.cat((x, input4x), dim=1))

        # 通道 MB
        x1s = self.CM_model1(input)
        x = self.cdown_conv1(x1s)
        x2s = self.CM_model2(torch.cat((x, input2x), dim=1))
        x = self.cdown_conv2(x2s)
        x3s = self.CM_model3(torch.cat((x, input4x), dim=1))

        # MS-MOE
        f3 = self.msmoe(x1c, x2c, x3c, x1s, x2s, x3s)

        # 上采样
        up1 = self.upsample_2x1(f3)
        up2 = self.upsample_2x2(up1)
        output = self.conv(up2)
        del x1c, x2c, x3c, x1s, x2s, x3s

        return output


if __name__ == '__main__':
    # 创建模型
    model = Y_MBNet1(in_channels=1, him_channels=2, out_channels=1)  # him_channels=2,3,4,5
    # 创建模拟数据
    input_data = torch.randn(10, 1, 51, 51)
    # 模型前向传播
    output_data = model(input_data)
    # 打印输出数据形状
    print("Output1 Shape:", output_data.shape)

