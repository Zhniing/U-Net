from blocks import *
import torch
import torch.nn.functional as F


class Unet(nn.Module):
    """瑕疵品,共用卷积块导致参数共享,导致效果很差"""
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
        x3 = self.conv_256(x3)  # 不能这样写，这样就是共享参数了

        x4 = self.maxpool(x3)
        x4 = self.conv_256_512(x4)
        x4 = self.conv_512(x4)

        # Expanding path
        d3 = self.up(x4)
        d3 = self.conv_512_256(d3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.conv_512_256(d3)
        d3 = self.conv_256(d3)  # 不能这样写，这样就是共享参数了

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
    """基础U-Net"""
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


class Unet3(nn.Module):
    """多模态U-Net"""
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        # t1 encoder block
        self.l1_encoder_t1 = nn.Sequential(
            ConvBlock2d(in_ch, 64),
            ConvBlock2d(64, 64)
        )

        self.l2_encoder_t1 = nn.Sequential(
            ConvBlock2d(64, 128),
            ConvBlock2d(128, 128)
        )

        self.l3_encoder_t1 = nn.Sequential(
            ConvBlock2d(128, 256),
            ConvBlock2d(256, 256)
        )

        self.l4_t1 = nn.Sequential(
            ConvBlock2d(256, 512),
            ConvBlock2d(512, 512)
        )

        # t2 encoder block
        self.l1_encoder_t2 = nn.Sequential(
            ConvBlock2d(in_ch, 64),
            ConvBlock2d(64, 64)
        )

        self.l2_encoder_t2 = nn.Sequential(
            ConvBlock2d(64, 128),
            ConvBlock2d(128, 128)
        )

        self.l3_encoder_t2 = nn.Sequential(
            ConvBlock2d(128, 256),
            ConvBlock2d(256, 256)
        )

        self.l4_t2 = nn.Sequential(
            ConvBlock2d(256, 512),
            ConvBlock2d(512, 512)
        )

        # self.l1_fuse = ConvBlock2d(128, 64)
        # self.l2_fuse = ConvBlock2d(256, 128)
        # self.l3_fuse = ConvBlock2d(512, 256)
        # self.l4_fuse = ConvBlock2d(1024, 512)
        self.l1_fuse = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.l2_fuse = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.l3_fuse = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.l4_fuse = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)

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

    def forward(self, t1, t2):
        # encoder path: t1
        t11 = self.l1_encoder_t1(t1)  # 两条路共享权值
        t12 = self.l2_encoder_t1(self.down(t11))
        t13 = self.l3_encoder_t1(self.down(t12))

        t14 = self.l4_t1(self.down(t13))

        # encoder path: t2
        t21 = self.l1_encoder_t2(t2)
        t22 = self.l2_encoder_t2(self.down(t21))
        t23 = self.l3_encoder_t2(self.down(t22))

        t24 = self.l4_t2(self.down(t23))

        x1 = self.l1_fuse(torch.cat((t11, t21), dim=1))
        x2 = self.l2_fuse(torch.cat((t12, t22), dim=1))
        x3 = self.l3_fuse(torch.cat((t13, t23), dim=1))
        x4 = self.l4_fuse(torch.cat((t14, t24), dim=1))
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


class MMAN(nn.Module):
    """Multi-modality aggregation network"""
    def __init__(self, in_ch, out_ch):
        super(MMAN, self).__init__()

        self.dib_t1_1 = DIB(in_ch=in_ch, out_ch=32)
        self.dib_t1_2 = DIB(in_ch=32, out_ch=32)
        self.dib_t1_3 = DIB(in_ch=32, out_ch=64)
        self.dib_t1_4 = DIB(in_ch=64, out_ch=64)

        self.dib_t2_1 = DIB(in_ch=in_ch, out_ch=32)
        self.dib_t2_2 = DIB(in_ch=32, out_ch=32)
        self.dib_t2_3 = DIB(in_ch=32, out_ch=64)
        self.dib_t2_4 = DIB(in_ch=64, out_ch=64)

        self.dib_5 = DIB(in_ch=2*32, out_ch=16)
        self.dib_6 = DIB(in_ch=2*32, out_ch=16)
        self.dib_7 = DIB(in_ch=2*64, out_ch=16)
        self.dib_8 = DIB(in_ch=2*64, out_ch=16)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_1x1 = nn.Conv2d(in_channels=4*16, out_channels=out_ch, kernel_size=3, padding=1)

    def forward(self, t1, t2):
        t1_1 = self.dib_t1_1(t1)
        t1_2 = self.dib_t1_2(self.maxpool(t1_1))
        t1_3 = self.dib_t1_3(self.maxpool(t1_2))
        t1_4 = self.dib_t1_4(self.maxpool(t1_3))

        t2_1 = self.dib_t2_1(t2)
        t2_2 = self.dib_t2_2(self.maxpool(t2_1))
        t2_3 = self.dib_t2_3(self.maxpool(t2_2))
        t2_4 = self.dib_t2_4(self.maxpool(t2_3))

        c1 = torch.cat((t1_1, t2_1), dim=1)
        c2 = torch.cat((t1_2, t2_2), dim=1)
        c3 = torch.cat((t1_3, t2_3), dim=1)
        c4 = torch.cat((t1_4, t2_4), dim=1)

        F1 = self.dib_5(c1)
        F2 = self.up_2(self.dib_6(c2))
        F3 = self.up_4(self.dib_7(c3))
        F4 = self.up_8(self.dib_8(c4))

        F1234 = torch.cat((F1, F2, F3, F4), dim=1)
        # torch.cat((F4, F3, F2, F1), dim=1)

        output = self.conv_1x1(F1234)
        predict = F.softmax(output, dim=1)

        return predict
