import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class AP_MP(nn.Module):
    def __init__(self,stride=2):
        super(AP_MP,self).__init__()
        self.sz=stride
        self.gapLayer=nn.AvgPool2d(kernel_size=self.sz,stride=self.sz)
        self.gmpLayer=nn.MaxPool2d(kernel_size=self.sz,stride=self.sz)
    def forward(self,x1,x2):
        apimg=self.gapLayer(x1)
        mpimg=self.gmpLayer(x2)
        K  = abs(apimg-mpimg)
        byimg=torch.norm(K,p=2,dim=1,keepdim=True)
        return byimg

class DLEF(nn.Module):
    def __init__(self,channel):
        super(DLEF,self).__init__()
        self.channel=channel
        self.conv1=BasicConv2d(channel,channel,3,padding=1)
        self.conv2=BasicConv2d(channel,channel,3,padding=1)
        self.CA1=ChannelAttention(self.channel)
        self.CA2=ChannelAttention(self.channel)
        self.SA1=SpatialAttention()
        self.SA2=SpatialAttention()
        self.glbamp=AP_MP()
        self.conv=BasicConv2d(channel*2+1,channel,kernel_size=1,stride=1)
        self.upsample2 = nn.Upsample(scale_factor=2 , mode='bilinear', align_corners=True)
        self.upSA=SpatialAttention()
    def forward(self,x,up):
        x1=self.conv1(x)
        x2=self.conv2(x)
        if(torch.is_tensor(up)):
            C = up
            D = self.upSA(C)
            x2 = x2 * D + x2
        F = self.CA1(x1)
        x1=x1+x1*F
        G = self.CA2(x2)
        x2=x2+x2*G
        H = self.SA2(x2)
        nx1=x1+x1*H
        I = self.SA1(x1)
        nx2=x2+x2*I
        A = self.glbamp(nx1,nx2)
        gamp=self.upsample2(A)
        B = torch.cat([nx1,gamp,nx2],dim=1)
        res=self.conv(B)
        return res