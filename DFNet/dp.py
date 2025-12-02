import torch
from torch import nn
import torch.nn.functional as F

class GDPN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid())
        self.calibration = nn.Sequential(nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.noise_suppress = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid())
        self.edge_enhance = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.pyramid_pools = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in [2, 4, 8]])
        self.pyramid_convs = nn.ModuleList([nn.Conv2d(channels, channels//4, 1) for _ in range(3)])
        self.convert=nn.Conv2d(48,64,1)

    def forward(self, x):
        channel_att = self.se(x)
        x_channel = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial(spatial)
        x_spatial = x * spatial_att
        pyramid_feats = []
        for pool, conv in zip(self.pyramid_pools, self.pyramid_convs):
            p = pool(x)
            p = conv(p)
            p = F.interpolate(p, size=x.shape[2:], mode='bilinear', align_corners=True)
            pyramid_feats.append(p)
        pyramid_out = torch.cat(pyramid_feats, dim=1)
        combined = torch.cat([x_channel, x_spatial], dim=1)
        calibrated = self.calibration(combined)
        noise_mask = self.noise_suppress(calibrated)
        denoised = calibrated * noise_mask
        edge = self.edge_enhance(denoised)
        enhanced = edge - denoised
        pyramid_out=self.convert(pyramid_out)
        final_out = enhanced + pyramid_out
        return final_out