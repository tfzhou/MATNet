import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        resnet_im = models.resnet101(pretrained=True)
        self.conv1_1 = resnet_im.conv1
        self.bn1_1 = resnet_im.bn1
        self.relu_1 = resnet_im.relu
        self.maxpool_1 = resnet_im.maxpool

        self.res2_1 = resnet_im.layer1
        self.res3_1 = resnet_im.layer2
        self.res4_1 = resnet_im.layer3
        self.res5_1 = resnet_im.layer4

        resnet_fl = models.resnet101(pretrained=True)
        self.conv1_2 = resnet_fl.conv1
        self.bn1_2 = resnet_fl.bn1
        self.relu_2 = resnet_fl.relu
        self.maxpool_2 = resnet_fl.maxpool

        self.res2_2 = resnet_fl.layer1
        self.res3_2 = resnet_fl.layer2
        self.res4_2 = resnet_fl.layer3
        self.res5_2 = resnet_fl.layer4

        self.gated_res2 = Gated(256*2)
        self.gated_res3 = Gated(512*2)
        self.gated_res4 = Gated(1024*2)
        self.gated_res5 = Gated(2048*2)

        self.coa_res3 = CoAttention(channel=512)
        self.coa_res4 = CoAttention(channel=1024)
        self.coa_res5 = CoAttention(channel=2048)

    def forward_res2(self, f1, f2):
        x1 = self.conv1_1(f1)
        x1 = self.bn1_1(x1)
        x1 = self.relu_1(x1)
        x1 = self.maxpool_1(x1)
        r2_1 = self.res2_1(x1)

        x2 = self.conv1_2(f2)
        x2 = self.bn1_2(x2)
        x2 = self.relu_2(x2)
        x2 = self.maxpool_2(x2)
        r2_2 = self.res2_2(x2)

        return r2_1, r2_2

    def forward(self, f1, f2):
        r2_1, r2_2 = self.forward_res2(f1, f2)
        r2 = torch.cat([r2_1, r2_2], dim=1)

        # res3
        r3_1 = self.res3_1(r2_1)
        r3_2 = self.res3_2(r2_2)

        Za, Zb, Qa, Qb = self.coa_res3(r3_1, r3_2)
        r3_1 = F.relu(Zb + r3_1)
        r3_2 = F.relu(Qb + r3_2)
        r3 = torch.cat([r3_1, r3_2], dim=1)

        # res4
        r4_1 = self.res4_1(r3_1)
        r4_2 = self.res4_2(r3_2)

        Za, Zb, Qa, Qb = self.coa_res4(r4_1, r4_2)
        r4_1 = F.relu(Zb + r4_1)
        r4_2 = F.relu(Qb + r4_2)
        r4 = torch.cat([r4_1, r4_2], dim=1)

        # res5
        r5_1 = self.res5_1(r4_1)
        r5_2 = self.res5_2(r4_2)

        Za, Zb, Qa, Qb = self.coa_res5(r5_1, r5_2)
        r5_1 = F.relu(Zb + r5_1)
        r5_2 = F.relu(Qb + r5_2)
        r5 = torch.cat([r5_1, r5_2], dim=1)

        r5_gated = self.gated_res5(r5)
        r4_gated = self.gated_res4(r4)
        r3_gated = self.gated_res3(r3)
        r2_gated = self.gated_res2(r2)

        return r5_gated, r4_gated, r3_gated, r2_gated


class CoAttention(nn.Module):
    def __init__(self, channel):
        super(CoAttention, self).__init__()

        d = channel // 16
        self.proja = nn.Conv2d(channel, d, kernel_size=1)
        self.projb = nn.Conv2d(channel, d, kernel_size=1)

        self.bottolneck1 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                )

        self.bottolneck2 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                )

        self.proj1 = nn.Conv2d(channel, 1, kernel_size=1)
        self.proj2 = nn.Conv2d(channel, 1, kernel_size=1)

        self.bna = nn.BatchNorm2d(channel)
        self.bnb = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, Qa, Qb):
        # cascade 1
        Qa_1, Qb_1 = self.forward_sa(Qa, Qb)
        _, Zb = self.forward_co(Qa_1, Qb_1)

        Pa = F.relu(Zb + Qa)
        Pb = F.relu(Qb_1 + Qb)

        # cascade 2
        Qa_2, Qb_2 = self.forward_sa(Pa, Pb)
        _, Zb = self.forward_co(Qa_2, Qb_2)

        Pa = F.relu(Zb + Pa)
        Pb = F.relu(Qb_2 + Pb)

        # cascade 3
        Qa_3, Qb_3 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_3, Qb_3)

        Pa = F.relu(Zb + Pa)
        Pb = F.relu(Qb_3 + Pb)

        # cascade 4
        Qa_4, Qb_4 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_4, Qb_4)

        Pa = F.relu(Zb + Pa)
        Pb = F.relu(Qb_4 + Pb)

        # cascade 5
        Qa_5, Qb_5 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_5, Qb_5)

        return Za, Zb, Qa_5, Qb_5

    def forward_sa(self, Qa, Qb):
        Aa = self.proj1(Qa)
        Ab = self.proj2(Qb)

        n, c, h, w = Aa.shape
        Aa = Aa.view(-1, h*w)
        Ab = Ab.view(-1, h*w)

        Aa = F.softmax(Aa)
        Ab = F.softmax(Ab)

        Aa = Aa.view(n, c, h, w)
        Ab = Ab.view(n, c, h, w)

        Qa_attened = Aa * Qa
        Qb_attened = Ab * Qb

        return Qa_attened, Qb_attened

    def forward_co(self, Qa, Qb):
        Qa_low = self.proja(Qa)
        Qb_low = self.projb(Qb)

        N, C, H, W = Qa_low.shape
        Qa_low = Qa_low.view(N, C, H * W)
        Qb_low = Qb_low.view(N, C, H * W)
        Qb_low = torch.transpose(Qb_low, 1, 2)

        L = torch.bmm(Qb_low, Qa_low)

        Aa = F.tanh(L)
        Ab = torch.transpose(Aa, 1, 2)

        N, C, H, W = Qa.shape

        Qa_ = Qa.view(N, C, H * W)
        Qb_ = Qb.view(N, C, H * W)

        Za = torch.bmm(Qb_, Aa)
        Zb = torch.bmm(Qa_, Ab)
        Za = Za.view(N, C, H, W)
        Zb = Zb.view(N, C, H, W)

        Za = F.normalize(Za)
        Zb = F.normalize(Zb)

        return Za, Zb


class Gated(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Gated, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.excitation_1 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True))

        self.excitation_2 = nn.Sequential(
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

        self.global_attention = nn.Sequential(
            nn.Linear(channel // reduction, 1),
            nn.Sigmoid()
        )

        kernel_size = 7
        self.spatial = BasicConv(1, 1, kernel_size, stride=1,
                                 padding=(kernel_size-1) // 2, relu=False)

    def forward(self, U):
        # se layer
        b, c, h, w = U.shape
        S = self.avg_pool(U).view(b, c)
        E_1 = self.excitation_1(S)

        E_local = self.excitation_2(E_1).view(b, c, 1, 1)
        U_se = E_local * U

        # spatial layer
        U_se_max = torch.max(U_se, 1)[0].unsqueeze(1)
        SP_Att = self.spatial(U_se_max)
        U_se_sp = SP_Att * U_se

        # global layer
        E_global = self.global_attention(E_1).view(b, 1, 1, 1)
        V = E_global * U_se_sp

        # residual layer
        O = U + V

        return O

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BoundaryModule(nn.Module):
    def __init__(self, inchannel):
        super(BoundaryModule, self).__init__()

        self.bn1 = nn.BatchNorm2d(inchannel)
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1,
                               padding=1)
        self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        mdim = 256
        self.GC = GC(4096+1, mdim)
        self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.RF4 = Refine(2048+1, mdim)
        self.RF3 = Refine(1024+1, mdim)
        self.RF2 = Refine(512+1, mdim)

        self.pred5 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred4 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred3 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.concat = nn.Conv2d(4, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.bdry5 = BoundaryModule(4096)
        self.bdry4 = BoundaryModule(2048)
        self.bdry3 = BoundaryModule(1024)
        self.bdry2 = BoundaryModule(512)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, r5, r4, r3, r2):
        p5 = self.bdry5(r5)
        p4 = self.bdry4(r4)
        p3 = self.bdry3(r3)
        p2 = self.bdry2(r2)

        p2_up = F.interpolate(p2, size=(473, 473), mode='bilinear')
        p3_up = F.interpolate(p3, size=(473, 473), mode='bilinear')
        p4_up = F.interpolate(p4, size=(473, 473), mode='bilinear')
        p5_up = F.interpolate(p5, size=(473, 473), mode='bilinear')

        concat = torch.cat([p2_up, p3_up, p4_up, p5_up], dim=1)
        p = self.concat(concat)

        p2_up = torch.sigmoid(p2_up)
        p3_up = torch.sigmoid(p3_up)
        p4_up = torch.sigmoid(p4_up)
        p5_up = torch.sigmoid(p5_up)
        p = torch.sigmoid(p)

        r5 = torch.cat((r5, p5), dim=1)
        r4 = torch.cat((r4, p4), dim=1)
        r3 = torch.cat((r3, p3), dim=1)
        r2 = torch.cat((r2, p2), dim=1)

        m = self.forward_mask(r5, r4, r3, r2)

        return m, p, p2_up, p3_up, p4_up, p5_up

    def forward_mask(self, x, r4, r3, r2):
        x = self.GC(x)
        r = self.convG1(F.relu(x))
        r = self.convG2(F.relu(r))
        m5 = x + r
        m4 = self.RF4(r4, m5)
        m3 = self.RF3(r3, m4)
        m2 = self.RF2(r2, m3)

        p2 = self.pred2(F.relu(m2))
        p2_up = F.interpolate(p2, size=(473, 473), mode='bilinear')
        p2_s = torch.sigmoid(p2_up)

        return p2_s


class GC(nn.Module):
    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))
        self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x


class AtrousBlock(nn.Module):
    def __init__(self, inplanes, planes, rate, stride=1):
        super(AtrousBlock, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                              dilation=rate, padding=rate)

    def forward(self, x):
        return self.conv(x)


class PyramidDilationConv(nn.Module):
    def __init__(self, inplanes, planes):
        super(PyramidDilationConv, self).__init__()

        rate = [3, 5, 7]

        self.block0 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.block1 = AtrousBlock(inplanes, planes, rate[0])
        self.block2 = AtrousBlock(inplanes, planes, rate[1])
        self.block3 = AtrousBlock(inplanes, planes, rate[2])
        self.bn = nn.BatchNorm2d(planes*4)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x)
        x2 = self.block1(x)
        x3 = self.block1(x)

        xx = torch.cat([x0, x1, x2, x3], dim=1)
        xx = self.bn(xx)
        return xx


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

        outplanes = int(planes / 4)
        self.pdc = PyramidDilationConv(inplanes, outplanes)

    def forward(self, f, pm):
        s = self.pdc(f)
        sr = self.convFS2(F.relu(s))
        sr = self.convFS3(F.relu(sr))
        s = s + sr

        m = s + F.interpolate(pm, size=s.shape[2:4], mode='bilinear')

        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m
