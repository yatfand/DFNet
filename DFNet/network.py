import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_
import timm
from Backbone.PVTv2 import PyramidVisionTransformerV2
from functools import partial
from cd import AttFusion
from ef import DLEF
from dp import GDPN

class SA(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(SA, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]
        return F.relu(w * out1 + b, inplace=True)

class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,scale_learnable=True, bias_learnable=True,mode=None, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(torch.empty(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(torch.empty(1), requires_grad=bias_learnable)
        self.scale_value = scale_value
        self.bias_value = bias_value
        self._initialized = False
    def forward(self, x):
        if not self._initialized:
            device = x.device
            self.scale.data = (self.scale_value * torch.ones(1, device=device))
            self.bias.data = (self.bias_value * torch.ones(1, device=device))
            self._initialized = True
        return self.scale * self.relu(x) ** 2 + self.bias

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,bias=False,  **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class DDDF(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,act1_layer=StarReLU,
                 bias=False, num_filters=4, size=14,**kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim = dim, mlp_ratio = reweight_expansion_ratio, out_features = num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(torch.randn(self.size, self.filter_size, num_filters, 2, dtype=torch.float32) * 0.02)
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)
    def forward(self, rgb, dep):
        rgb = rgb.permute(0, 2, 3, 1)
        dep = dep.permute(0, 2, 3, 1)
        x = dep
        B, H, W, _ = x.shape
        D = x.mean(dim=(1, 2))
        A = self.reweight(D).view(B, self.num_filters,-1)
        routeing = A.softmax(dim=1)
        routeing = routeing.to(torch.complex64)
        rgb = self.pwconv1(rgb)
        rgb = self.act1(rgb)
        rgb = rgb.to(torch.float32)
        rgb = torch.fft.rfft2(rgb, dim=(1, 2), norm='ortho')
        complex_weights = resize_complex_weight(self.complex_weights, rgb.shape[1], rgb.shape[2])
        complex_weights = torch.view_as_complex(complex_weights.contiguous()) #复数化
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        weight = weight.view(-1, rgb.shape[1], rgb.shape[2], self.med_channels)
        x = rgb * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return x

def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    A = torch.nn.functional.interpolate(origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True)
    new_weight = A.permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight

class CMAT(nn.Module):
    def __init__(self, in_channel, K = 6):
        super(CMAT, self).__init__()
        self.sa1 = SA(in_channel)
        self.sa2 = SA(in_channel)
        if K == 1 or K == 2:
            self.att1 = DLEF(256)
            self.att2 = DLEF(256)
        elif K == 3:
            self.att1 = DDDF(dim=256, expansion_ratio=2, reweight_expansion_ratio=.25,
                                      act1_layer=StarReLU, bias=False, num_filters=4,
                                      size=16)
            self.att2 = DDDF(dim=256, expansion_ratio=2, reweight_expansion_ratio=.25,
                                      act1_layer=StarReLU,  bias=False, num_filters=4,
                                      size=16)
        elif K == 4:
            self.att1 = DDDF(dim=256, expansion_ratio=2, reweight_expansion_ratio=.25,
                                      act1_layer=StarReLU,  bias=False, num_filters=4,
                                      size=8)
            self.att2 = DDDF(dim=256, expansion_ratio=2, reweight_expansion_ratio=.25,
                                      act1_layer=StarReLU,  bias=False, num_filters=4,
                                      size=8)
        else:
            print("k错误")

    def forward(self, rgb, depth):
        rgb = self.sa1(rgb)
        depth = self.sa2(depth)
        feat_1 = self.att1(rgb, depth)
        feat_2 = self.att2(depth, rgb)
        out1 = rgb + feat_1
        out2 = depth + feat_2
        return out1, out2
    
class Fusion(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(Fusion, self).__init__()
        self.conv0 = nn.Conv2d(in_channel*2, 256, 3, 1, 1)
        self.bn0 = norm_layer(256)
    def forward(self, x1, x2):
        out1 = x1
        out2 = x1 * x2
        out  = torch.cat((out1, out2), dim=1)
        out = F.relu(self.bn0(self.conv0(out)), inplace=True)
        return out
class Segment(nn.Module):
    def __init__(self, backbone='abcd', aux_layers=True):
        super(Segment, self).__init__()
        self.aux_layers = aux_layers
        self.backbone = backbone
        if self.backbone == 'pvtv2':
            channels = [64, 128, 320, 512]
            self.backbone_rgb = PyramidVisionTransformerV2(patch_size=4,embed_dims=[64, 128, 320, 512],num_heads=[1, 2, 5, 8],
                        mlp_ratios=[8, 8, 4, 4],qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),depths=[3, 4, 6, 3],
                        sr_ratios=[8, 4, 2, 1],num_classes=0,in_chans=3)
            state_dict = torch.load("./pvt_v2_b2.pth")
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
            self.backbone_rgb.load_state_dict(state_dict)
            
            self.backbone_d = PyramidVisionTransformerV2(patch_size=4,embed_dims=[64, 128, 320, 512],num_heads=[1, 2, 5, 8],
                        mlp_ratios=[8, 8, 4, 4],qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),depths=[3, 4, 6, 3],
                        sr_ratios=[8, 4, 2, 1],num_classes=0,in_chans=1)
            d_state_dict1 = torch.load("./pvt_v2_b2.pth")
            d_state_dict1 = {k: v for k, v in d_state_dict1.items() if not k.startswith('head.')}
            if 'patch_embed1.proj.weight' in d_state_dict1:
                conv1_weight = d_state_dict1['patch_embed1.proj.weight']
                conv1_weight = conv1_weight.mean(dim=1, keepdim=True)
                d_state_dict1['patch_embed1.proj.weight'] = conv1_weight
            self.backbone_d.load_state_dict(d_state_dict1, strict=False)
        else:
            print(f'! self.backbone = {self.backbone}')

        self.cmat5 = CMAT(channels[3], K=4)
        self.cmat4 = CMAT(channels[2], K=3)
        self.cmat3 = CMAT(channels[1], K=2)
        self.cmat2 = CMAT(channels[0], K=1)
        self.fam54_1 = AttFusion(256, 256, xuhao=1)
        self.fam43_1 = AttFusion(256, 256, xuhao=2)
        self.fam32_1 = AttFusion(256, 256, xuhao=3)
        self.fam54_2 = AttFusion(256, 256, xuhao=1)
        self.fam43_2 = AttFusion(256, 256, xuhao=2)
        self.fam32_2 = AttFusion(256, 256, xuhao=3)
        self.fam21_1 = AttFusion(256, 256, xuhao=4)
        self.fusion = Fusion(256)
        if self.aux_layers:
            self.linear5_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear4_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear3_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear2_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear1_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear5_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear4_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear3_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear2_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear_out = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.SconvR2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.SconvR3 = nn.Conv2d(64, 320, kernel_size=3, stride=1, padding=1)
        self.SconvR4 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.SconvD2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.SconvD3 = nn.Conv2d(64, 320, kernel_size=3, stride=1, padding=1)
        self.SconvD4 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.SconvR5 = nn.Conv2d(128, 320, kernel_size=3, stride=1, padding=1)
        self.SconvR6 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1)
        self.SconvR7 = nn.Conv2d(320, 512, kernel_size=3, stride=1, padding=1)
        self.SconvD5 = nn.Conv2d(128, 320, kernel_size=3, stride=1, padding=1)
        self.SconvD6 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1)
        self.SconvD7 = nn.Conv2d(320, 512, kernel_size=3, stride=1, padding=1)
        self.GDPN_1 = GDPN(64)
        self.ref_conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.ref_bn1_1 = nn.BatchNorm2d(64)
        self.ref_relu_1 = nn.ReLU(inplace=True)
        self.ref_maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.ref_conv2_1 = nn.Conv2d(64, 256, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, rgb, depth):
        raw_size = rgb.size()[2:]
        if self.backbone == 'resnet50' or self.backbone == 'pvtv2':
            enc2_1, enc3_1, enc4_1, enc5_1 = self.backbone_rgb(rgb)
            enc2_2, enc3_2, enc4_2, enc5_2 = self.backbone_d(depth)
        enc23_1 = self.SconvR2(F.interpolate(enc2_1, size=enc3_1.size()[2:], mode='bilinear'))
        enc24_1 = self.SconvR3(F.interpolate(enc2_1, size=enc4_1.size()[2:], mode='bilinear'))
        enc25_1 = self.SconvR4(F.interpolate(enc2_1, size=enc5_1.size()[2:], mode='bilinear'))
        enc34_1 = self.SconvR5(F.interpolate(enc3_1, size=enc4_1.size()[2:], mode='bilinear'))
        enc35_1 = self.SconvR6(F.interpolate(enc3_1, size=enc5_1.size()[2:], mode='bilinear'))
        enc45_1 = self.SconvR7(F.interpolate(enc4_1, size=enc5_1.size()[2:], mode='bilinear'))
        enc23_2 = self.SconvD2(F.interpolate(enc2_2, size=enc3_2.size()[2:], mode='bilinear'))
        enc24_2 = self.SconvD3(F.interpolate(enc2_2, size=enc4_2.size()[2:], mode='bilinear'))
        enc25_2 = self.SconvD4(F.interpolate(enc2_2, size=enc5_2.size()[2:], mode='bilinear'))
        enc34_2 = self.SconvD5(F.interpolate(enc3_2, size=enc4_2.size()[2:], mode='bilinear'))
        enc35_2 = self.SconvD6(F.interpolate(enc3_2, size=enc5_2.size()[2:], mode='bilinear'))
        enc45_2 = self.SconvD7(F.interpolate(enc4_2, size=enc5_2.size()[2:], mode='bilinear'))
        A2 = enc2_1
        A3 = enc23_1 + enc3_1
        A4 = enc24_1 + enc34_1 + enc4_1
        A5 = enc25_1 + enc35_1 + enc45_1 + enc5_1
        B2 = enc2_2
        B3 = enc23_2 + enc3_2
        B4 = enc24_2 + enc34_2 + enc4_2
        B5 = enc25_2 + enc35_2 + enc45_2 + enc5_2
        de2_1, de2_2 = self.cmat2(A2, B2)
        de3_1, de3_2 = self.cmat3(A3, B3)
        de4_1, de4_2 = self.cmat4(A4, B4)
        out5_1, out5_2 = self.cmat5(A5, B5)
        out4_1 = self.fam54_1(de4_1, out5_1,xuhao=1)
        out3_1 = self.fam43_1(de3_1, out4_1,xuhao=2)
        out2_1 = self.fam32_1(de2_1, out3_1, xuhao=3)
        out4_2 = self.fam54_2(de4_2, out5_2,xuhao=1)
        out3_2 = self.fam43_2(de3_2, out4_2,xuhao=2)
        out2_2 = self.fam32_2(de2_2, out3_2, xuhao=3)
        x = self.ref_conv_1(rgb)
        x = self.ref_bn1_1(x)
        x = self.ref_relu_1(x)
        x = self.ref_maxpool_1(x)
        T1 = self.GDPN_1(x)
        T1 = self.ref_conv2_1(T1)
        T1 = F.interpolate(T1, size=out2_1.size()[2:], mode='bilinear')
        out1_1 = self.fam21_1(T1, out2_1, xuhao=4)
        out = self.fusion(out1_1, out2_2)
        out = self.linear_out(out)
        out = F.interpolate(out, size=raw_size, mode='bilinear', )
        if self.training and self.aux_layers:
            out5_1 = F.interpolate(self.linear5_1(out5_1), size=raw_size, mode='bilinear')
            out4_1 = F.interpolate(self.linear4_1(out4_1), size=raw_size, mode='bilinear')
            out3_1 = F.interpolate(self.linear3_1(out3_1), size=raw_size, mode='bilinear')
            out2_1 = F.interpolate(self.linear2_1(out2_1), size=raw_size, mode='bilinear')
            out1_1 = F.interpolate(self.linear1_1(out1_1), size=raw_size, mode='bilinear')
            out5_2 = F.interpolate(self.linear5_2(out5_2), size=raw_size, mode='bilinear')
            out4_2 = F.interpolate(self.linear4_2(out4_2), size=raw_size, mode='bilinear')
            out3_2 = F.interpolate(self.linear3_2(out3_2), size=raw_size, mode='bilinear')
            out2_2 = F.interpolate(self.linear2_2(out2_2), size=raw_size, mode='bilinear')
            return out, out1_1, out2_1, out3_1, out4_1, out5_1, out2_2, out3_2, out4_2, out5_2
        else:
            return out