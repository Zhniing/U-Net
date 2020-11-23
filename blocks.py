import torch
import torch.nn as nn


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock2d, self).__init__()

        # 定义网络模块
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),  # 尺寸不变
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)  # 原地(in-place)操作: 不占用额外空间
        )

    # 定义网络结构
    def forward(self, x):
        x = self.conv(x)
        return x


class DIB(nn.Module):
    """Dilated-Inception block"""
    def __init__(self, in_ch, out_ch):
        super(DIB, self).__init__()

        self.dil_conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1),  # dilation默认就为1，即普通卷积
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dil_conv_2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dil_conv_4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv_1x1 = nn.Conv2d(3 * out_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.dil_conv_1(x)
        x2 = self.dil_conv_2(x)
        x3 = self.dil_conv_4(x)

        x123 = torch.cat((x1, x2, x3), dim=1)

        output = self.conv_1x1(x123)

        return output
