import torch.nn as nn

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock2d, self).__init__()

        # 定义网络模块
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) # 原地(in-place)操作: 不占用额外空间
        )

    # 定义网络结构
    def forward(self, x):
        x = self.conv(x)
        return x
