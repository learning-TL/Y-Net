import torch
import torch.nn as nn
import numpy as np
import scipy.linalg
import torch.nn.init as init
import torch.nn.functional as F

def conv_upsample(size=(25, 25)):
    upsample = nn.Upsample(size=size, mode='bilinear', align_corners=True)
    return upsample

class GateNetwork(nn.Module):
    def __init__(self, input_size, num_experts, top_k):
        super(GateNetwork, self).__init__()
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.fc0 = nn.Linear(input_size, num_experts)
        self.fc1 = nn.Linear(input_size, num_experts)
        self.relu1 = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        init.zeros_(self.fc1.weight)
        self.sp = nn.Softplus()

    def forward(self, x):
        # Flatten the input tensor
        x = self.gap(x) + self.gap2(x)
        x = x.view(-1, self.input_size)
        inp = x
        # Pass the input through the gate network layers
        x = self.fc1(x)
        x = self.relu1(x)
        noise = self.sp(self.fc0(inp))
        noise_mean = torch.mean(noise, dim=1)
        noise_mean = noise_mean.view(-1, 1)
        std = torch.std(noise, dim=1)
        std = std.view(-1, 1)
        noram_noise = (noise - noise_mean) / std
        # Apply topK operation to get the highest K values and indices along dimension 1 (columns)
        topk_values, topk_indices = torch.topk(x + noram_noise, k=self.top_k, dim=1)

        # Set all non-topK values to -inf to ensure they are not selected by softmax
        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)
        x[~mask.bool()] = float('-inf')

        # Pass the masked tensor through softmax to get gating coefficients for each expert network
        gating_coeffs = self.softmax(x)

        return gating_coeffs


class ConvProce(nn.Module):
    def __init__(self, channel):
        super(ConvProce, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x):
        res = x
        x = self.relu1(self.conv1(x))
        x = self.conv2(x) + res
        return x


class MOE(nn.Module):
    def __init__(self, channels, k):
        super(MOE, self).__init__()
        self.num_experts = channels
        self.gate = GateNetwork(channels, channels, k)
        self.expert_networks_d = nn.ModuleList(
            [ConvProce(channels) for i in range(channels)])
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU()
        )

    def forward(self, x):
        cof = self.gate(x)
        out = torch.zeros_like(x).to(x.device)
        for idx in range(self.num_experts):
            if (cof[:, idx] == 0).all().item():
                continue
            mask = torch.where(cof[:, idx] > 0)[0]
            expert_layer = self.expert_networks_d[idx]
            expert_out = expert_layer(x[mask])
            cof_k = cof[mask, idx].view(-1, 1, 1, 1)
            out[mask] += expert_out * cof_k
        out = self.conv(out)
        return out


class MS_MOE(nn.Module):
    def __init__(self, num_channels):
        super(MS_MOE, self).__init__()
        # 2倍上采样
        self.upsample_2x = nn.Sequential(
            conv_upsample(size=(13, 13)),
            # nn.ConvTranspose2d(2 * num_channels + 2, 2 * num_channels + 2, kernel_size=3, stride=2, padding=1,
            #                    output_padding=1),
            nn.Conv2d(2 * num_channels + 2, 2 * num_channels + 2, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 上采样
        )
        # 4倍上采样
        self.upsample_4x = nn.Sequential(
            conv_upsample(size=(13, 13)),
            # nn.ConvTranspose2d(4 * num_channels + 2, 2 * num_channels + 2, kernel_size=3, stride=4, padding=1,
            #                    output_padding=3),
            nn.Conv2d( num_channels, num_channels, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.ReLU(),
            # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 上采样
        )
        self.moe = MOE(9 * num_channels, k=3)

    def forward(self, skip1, skip2, skip3):
        skip1 = self.upsample_4x(skip1)
        skip2 = self.upsample_2x(skip2)

        x = torch.cat((skip1, skip2, skip3), dim=1)
        out = self.moe(x)

        return out


class MS_MOE_SC(nn.Module):
    def __init__(self, num_channels):
        super(MS_MOE_SC, self).__init__()
        self.moe1 = MS_MOE(num_channels)
        self.moe2 = MS_MOE(num_channels)
    def forward(self, s_skip1, s_skip2, s_skip3, c_skip1, c_skip2, c_skip3):
        F_S = self.moe1(s_skip1, s_skip2, s_skip3)
        F_C = self.moe2(c_skip1, c_skip2, c_skip3)
        F = F_C + F_S
        return F


if __name__ == '__main__':
    # 创建模型
    model = MS_MOE_SC(num_channels=2)
    # 创建模拟数据
    input_data1 = torch.randn(10, 2, 51, 51)
    input_data2 = torch.randn(10, 6, 26, 26)
    input_data3 = torch.randn(10, 10, 13, 13)
    input_data4 = torch.randn(10, 2, 51, 51)
    input_data5 = torch.randn(10, 6, 26, 26)
    input_data6 = torch.randn(10, 10, 13, 13)
    # 模型前向传播
    output_data = model(input_data1, input_data2, input_data3, input_data4, input_data5,
                                                     input_data6)
    # torch.save(model, "E:/D/my_RRDB/save_model/model_test_SPAB.pt")  # 保存模型
    # 打印模型结构
    # print(model)
    # 打印输出数据形状
    print("Output1 Shape:", output_data.shape)
