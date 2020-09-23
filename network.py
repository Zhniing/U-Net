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


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()
        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]
            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C
            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)
        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss