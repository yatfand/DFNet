import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            print('执行了 if self.relu is not None')
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.mlp = nn.Sequential(Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types
    def forward(self, x, y):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return y * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        A = torch.max(x,1)[0].unsqueeze(1)
        B = torch.mean(x,1).unsqueeze(1)
        C = torch.cat((A, B), dim=1)
        return C

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x, y):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        A = y * scale
        return A

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()
    def forward(self, x, y):
        y_out = self.ChannelGate(x, y)
        y_out = self.SpatialGate(x, y_out)
        return y_out

class AttFusion(nn.Module):
    def __init__(self ,in_dim, out_dim,xuhao=5):
        super(AttFusion, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.reduc_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.rgb_d = CBAM(out_dim, 1)
        self.d_rgb = CBAM(out_dim, 1)
        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim), act_fn, )
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, rgb, depth,xuhao=5):
        if xuhao == 1 or xuhao == 2 or xuhao == 3:
            depth = self.upsample_2(depth)
        x_rgb = self.reduc_1(rgb)
        x_dep = depth
        x_rgb1 = self.layer_10(x_rgb)
        x_dep1 = self.layer_20(x_dep)
        x_dep_r = self.rgb_d(x_rgb1, x_dep1)
        x_rgb_r = self.d_rgb(x_dep1, x_rgb1)
        x_cat   = torch.cat((x_rgb_r, x_dep_r) ,dim=1)
        out1 = self.layer_ful1(x_cat)
        return out1