import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

YEARS        = range(2001, 2021)         # 2001…2020
MONTHS       = range(1, 13)              # Jan…Dec
# AREA         = pune_area                 # defined elsewhere
BATCH_SIZE   = 8
EPOCHS       = 25
LR           = 1e-4
BASE_ENC_CH  = 32
LC_EMBED_DIM = 8
LC_CLASSES   = 36
FUSE_BASE_CH = 64

# 4.1 Weight init
def kaiming_init(m):
    if isinstance(m,(nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)

# 4.2 BasicEncoder (LST & PM2.5)
class _ConvBNReLU(nn.Sequential):
    def __init__(self,in_c,out_c,k=3,s=1,p=1):
        super().__init__(
            nn.Conv2d(in_c,out_c,k,s,p,bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
class BasicEncoder(nn.Module):
    def __init__(self,in_ch,base=BASE_ENC_CH):
        super().__init__()
        self.b1=_ConvBNReLU(in_ch,base,k=3,s=1,p=1)
        self.b2=_ConvBNReLU(base,base*2,k=3,s=2,p=1)
        self.b3=_ConvBNReLU(base*2,base*4,k=3,s=2,p=1)
        self.out_channels=base*4
        self.apply(kaiming_init)
    def forward(self,x):
        x=self.b1(x); x=self.b2(x); return self.b3(x)

class LSTEncoder(BasicEncoder):
    def __init__(self): super().__init__(in_ch=1)
class PM25Encoder(BasicEncoder):
    def __init__(self): super().__init__(in_ch=1)

# --- Coordinate Convolution (CoordConv) ------------------
class CoordConv2d(nn.Module):
    """
    CoordConv layer: appends normalized X/Y coordinate channels
    before applying a Conv2d over (in_c + 2) → out_c.
    """
    def __init__(self, in_c, out_c, kernel_size, stride, padding, bias=False):
        super().__init__()
        # our conv sees 2 extra channels for X and Y
        self.conv = nn.Conv2d(in_channels=in_c + 2,
                              out_channels=out_c,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
    def forward(self, x):
        """
        x: tensor of shape (B, in_c, H, W)
        returns: tensor of shape (B, out_c, H_out, W_out)
        """
        b, _, h, w = x.shape
        # build coordinate maps in [-1, 1]
        xx = torch.linspace(-1, 1, w, device=x.device) \
                  .view(1, 1, 1, w).expand(b, 1, h, w)
        yy = torch.linspace(-1, 1, h, device=x.device) \
                  .view(1, 1, h, 1).expand(b, 1, h, w)
        x = torch.cat([x, xx, yy], dim=1)  # (B, in_c+2, H, W)
        return self.conv(x)

# --- Residual Dilated Block -------------------------------
class ResidualDilatedBlock(nn.Module):
    """
    A two-conv residual block with configurable dilation.
    Keeps spatial dims constant.
    """
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3,
                               padding=dilation,
                               dilation=dilation,
                               bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)

# --- CBAM Attention Module --------------------------------
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applies channel and spatial attention sequentially.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        # spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        avg = self.fc2(F.relu(self.fc1(self.avg_pool(x)), inplace=True))
        mx  = self.fc2(F.relu(self.fc1(self.max_pool(x)), inplace=True))
        x   = x * torch.sigmoid(avg + mx)
        # Spatial attention
        avg_c = x.mean(dim=1, keepdim=True)
        max_c, _ = x.max(dim=1, keepdim=True)
        x   = x * torch.sigmoid(self.conv_spatial(torch.cat([avg_c, max_c], dim=1)))
        return x

# --- LandCoverEncoder ------------------------------------
class LandCoverEncoder(nn.Module):
    """
    Encoder for 30 m land‑cover maps that:
      1. Embeds class IDs into vectors.
      2. Applies CoordConv to add positional info.
      3. Strided-convs to downsample to 1 km grid.
      4. Residual dilated block to enlarge receptive field.
      5. CBAM attention to reweight salient features.
    Input:  (B, 1, H30, W30)  integer class IDs in [0..n_classes].
    Output: (B, base*4, H30/4, W30/4) feature tensor at 1 km resolution.
    """
    def __init__(self, n_classes=36, embed_dim=8, base=32):
        super().__init__()
        self.base = base
        # 1) Class embedding → (B, H30, W30, embed_dim)
        self.embed = nn.Embedding(num_embeddings=n_classes + 1,
                                  embedding_dim=embed_dim,
                                  padding_idx=0)
        # 2) CoordConv down‑sample to H30/2, W30/2
        self.down1 = CoordConv2d(in_c=embed_dim,
                                 out_c=base,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.bn1   = nn.BatchNorm2d(base)
        self.act1  = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 3) Strided conv down‑sample to H30/4, W30/4
        self.down2 = nn.Conv2d(base,
                               base * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn2   = nn.BatchNorm2d(base * 2)
        self.act2  = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 4) Residual dilated block (keeps H30/4, W30/4)
        self.res_block = ResidualDilatedBlock(base * 2, dilation=2)

        # 5) Project to base*4 channels
        self.project = nn.Sequential(
            nn.Conv2d(base * 2,
                      base * 4,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True)
        )

        # 6) Channel + spatial attention
        self.cbam = CBAM(channels=base * 4)

        # record output channels for downstream fusion
        self.out_channels = base * 4

        # initialize weights
        self.apply(kaiming_init)

    def forward(self, x):
        # x: (B,1,H30,W30)
        b = x.squeeze(1).long()              # (B,H30,W30)
        x = self.embed(b)                    # (B,H30,W30,embed_dim)
        x = x.permute(0, 3, 1, 2).contiguous()# (B,embed_dim,H30,W30)

        x = self.act1(self.bn1(self.down1(x)))  # (B, base, H30/2, W30/2)
        x = self.act2(self.bn2(self.down2(x)))  # (B, base*2, H30/4, W30/4)
        x = self.res_block(x)                   # (B, base*2, H30/4, W30/4)
        x = self.project(x)                     # (B, base*4, H30/4, W30/4)
        x = self.cbam(x)                        # (B, base*4, H30/4, W30/4)
        return x

# 4.4 Attention‑U‑Net Fusion Head
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        # g: gating signal (from decoder), x: skip‐connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1, inplace=True)
        psi = self.psi(psi)
        return x * psi

class FusionAttentionUNet(nn.Module):
    def __init__(self, in_ch, base_ch=64, out_ch=1):
        super().__init__()
        # --- Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        # --- Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True)
        )

        # --- Attention Gates
        self.att2 = AttentionGate(F_g=base_ch*2, F_l=base_ch*2, F_int=base_ch)
        self.att1 = AttentionGate(F_g=base_ch,   F_l=base_ch,   F_int=base_ch//2)

        # --- Decoder Stage 1
        self.up1 = nn.Sequential(
            nn.Upsample(size=(5,5)),
            nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True)
        )

        # --- Decoder Stage 2
        self.up2 = nn.Sequential(
            nn.Upsample(size=(11,11)),
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        # --- Decoder Stage 3 (Adjusted)
        self.up3 = nn.Sequential(
            nn.Upsample(size=(20,21), mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),  # Adjusted from base_ch*2
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        # --- Final up-scale
        self.final_upscale = nn.Sequential(
            nn.Upsample(size=(41,43), mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch, out_ch, kernel_size=1)
        )

    def forward(self, x):
        # --- Encoder ---
        e1 = self.enc1(x)           # (B, base_ch, 11,11)
        p1 = self.pool1(e1)         # (B, base_ch, 5,5)
        e2 = self.enc2(p1)          # (B, base_ch*2, 5,5)
        p2 = self.pool2(e2)         # (B, base_ch*2, 2,2)

        # --- Bottleneck ---
        b = self.bottleneck(p2)     # (B, base_ch*4, 2,2)

        # --- Decoder Stage 1 ---
        d1 = self.up1(b)            # (B, base_ch*2, 5,5)
        e2_att = self.att2(g=d1, x=e2)
        d1 = torch.cat([d1, e2_att], dim=1)
        d1 = self.dec1(d1)          # (B, base_ch*2, 5,5)

        # --- Decoder Stage 2 ---
        d2 = self.up2(d1)           # (B, base_ch, 11,11)
        e1_att = self.att1(g=d2, x=e1)
        d2 = torch.cat([d2, e1_att], dim=1)
        d2 = self.dec2(d2)          # (B, base_ch, 11,11)

        # --- Decoder Stage 3 (Adjusted) ---
        d3 = self.up3(d2)           # (B, base_ch, 20,21)
        d3 = self.dec3(d3)          # (B, base_ch, 20,21) - No concatenation

        # --- Final up-scale ---
        out = self.final_upscale(d3)  # (B, out_ch, 41,43)
        return out