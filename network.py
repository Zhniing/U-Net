from blocks import *
import torch
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # 定义网络模块
        self.conv_in_64   = ConvBlock2d(in_ch=in_ch, out_ch=64)
        self.conv_64      = ConvBlock2d(64, 64)
        self.conv_64_128  = ConvBlock2d(64, 128)
        self.conv_128     = ConvBlock2d(128, 128)
        self.conv_128_256 = ConvBlock2d(128, 256)
        self.conv_256     = ConvBlock2d(256, 256)
        self.conv_256_512 = ConvBlock2d(256, 512)
        self.conv_512     = ConvBlock2d(512, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_512_256 = ConvBlock2d(in_ch=512, out_ch=256)
        self.conv_256_128 = ConvBlock2d(256, 128)
        self.conv_128_64  = ConvBlock2d(128, 64)

        self.conv_64_out_1x1 = nn.Conv2d(in_channels=64, out_channels=out_ch,
                                         kernel_size=1, stride=1, padding=0)

    # 定义网络结构（把模块串起来）
    def forward(self, x):
        # Contracting path
        x1 = self.conv_in_64(x)
        x1 = self.conv_64(x1)

        x2 = self.maxpool(x1)
        x2 = self.conv_64_128(x2)
        x2 = self.conv_128(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv_128_256(x3)
        x3 = self.conv_256(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv_256_512(x4)
        x4 = self.conv_512(x4)

        # Expanding path
        d3 = self.up(x4)
        d3 = self.conv_512_256(d3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.conv_512_256(d3)
        d3 = self.conv_256(d3)

        d2 = self.up(d3)
        d2 = self.conv_256_128(d2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.conv_256_128(d2)
        d2 = self.conv_128(d2)

        d1 = self.up(d2)
        d1 = self.conv_128_64(d1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.conv_128_64(d1)
        d1 = self.conv_64(d1)

        output = self.conv_64_out_1x1(d1)
        output = F.softmax(output, dim=1)

        return output


class Unet2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.l1_encoder = nn.Sequential(
            ConvBlock2d(in_ch, 64),
            ConvBlock2d(64, 64)
        )

        self.l2_encoder = nn.Sequential(
            ConvBlock2d(64, 128),
            ConvBlock2d(128, 128)
        )

        self.l3_encoder = nn.Sequential(
            ConvBlock2d(128, 256),
            ConvBlock2d(256, 256)
        )

        self.l4 = nn.Sequential(
            ConvBlock2d(256, 512),
            ConvBlock2d(512, 512)
        )

        self.up3 = nn.Sequential(
            self.up,
            ConvBlock2d(512, 256)
        )

        self.l3_decoder = nn.Sequential(
            ConvBlock2d(512, 256),
            ConvBlock2d(256, 256)
        )

        self.up2 = nn.Sequential(
            self.up,
            ConvBlock2d(256, 128)
        )

        self.l2_decoder = nn.Sequential(
            ConvBlock2d(256, 128),
            ConvBlock2d(128, 128)
        )

        self.up1 = nn.Sequential(
            self.up,
            ConvBlock2d(128, 64)
        )

        self.l1_decoder = nn.Sequential(
            ConvBlock2d(128, 64),
            ConvBlock2d(64, 64)
        )

        self.conv_1x1 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # encoder path
        x1 = self.l1_encoder(x)
        x2 = self.l2_encoder(self.down(x1))
        x3 = self.l3_encoder(self.down(x2))

        x4 = self.l4(self.down(x3))
        d4 = x4

        # decoder path
        d3 = torch.cat((x3, self.up3(d4)), dim=1)
        d3 = self.l3_decoder(d3)
        d2 = torch.cat((x2, self.up2(d3)), dim=1)
        d2 = self.l2_decoder(d2)
        d1 = torch.cat((x1, self.up1(d2)), dim=1)
        d1 = self.l1_decoder(d1)

        # output
        output = self.conv_1x1(d1)
        predict = F.softmax(output, dim=1)

        return predict
