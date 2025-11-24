# wcgan.py
# Conditional WGAN-GP with SPADE generator + Projection Discriminator
# Supports 64/128/256 outputs via num_upsamples

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# -----------------------------
# Blocks
# -----------------------------

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, embedding_dim, fixed_spatial_size=4):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta  = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.embedding_proj = nn.Linear(
            embedding_dim, label_nc * fixed_spatial_size * fixed_spatial_size
        )
        self.label_nc = label_nc
        self.fixed_spatial_size = fixed_spatial_size

    def forward(self, x, embedding):
        b, _, h, w = x.size()
        label_map = self.embedding_proj(embedding).view(
            b, self.label_nc, self.fixed_spatial_size, self.fixed_spatial_size
        )
        label_map = F.interpolate(label_map, size=(h, w), mode="nearest")
        x_norm = self.param_free_norm(x)
        actv = self.mlp_shared(label_map)
        gamma = self.mlp_gamma(actv)
        beta  = self.mlp_beta(actv)
        return x_norm * (1 + gamma) + beta


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, label_nc=16, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.norm1 = SPADE(in_channels,  label_nc, embedding_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = SPADE(out_channels, label_nc, embedding_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if in_channels != out_channels or upsample:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x, embedding):
        residual = x
        out = self.norm1(x, embedding)
        out = self.relu1(out)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.conv1(out)
        out = self.norm2(out, embedding)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.upsample:
            residual = F.interpolate(residual, scale_factor=2, mode="nearest")
        residual = self.skip_proj(residual)
        return out + residual


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim,      1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, h*w).permute(0,2,1)  # [B, HW, C/8]
        proj_key   = self.key_conv(x).view(b, -1, h*w)                   # [B, C/8, HW]
        energy = torch.bmm(proj_query, proj_key)                          # [B, HW, HW]
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(b, c, h*w)                   # [B, C, HW]
        out = torch.bmm(proj_value, attention.permute(0,2,1))             # [B, C, HW]
        out = out.view(b, c, h, w)
        return self.gamma * out + x

# -----------------------------
# Generator
# -----------------------------

class Generator(nn.Module):
    """
    output resolution = 4 * (2 ** num_upsamples)
    64x64: num_upsamples=4; 128x128: 5; 256x256: 6
    """
    def __init__(
        self,
        noise_dim: int = 128,
        audio_embed_dim: int = 512,
        out_channels: int = 3,
        base_channels: int = 64,
        num_upsamples: int = 4,
        attn_at: tuple = (32, 64),
        spade_label_nc: int = 16,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.audio_embed_dim = audio_embed_dim
        self.input_dim = noise_dim + audio_embed_dim

        # conditioning MLP (now used)
        self.audio_mlp = nn.Sequential(
            nn.Linear(audio_embed_dim, audio_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(audio_embed_dim, audio_embed_dim),
        )

        self.base_channels = base_channels
        self.num_upsamples = num_upsamples
        self.attn_at = set(attn_at)

        start_ch = base_channels * 8
        self.fc = nn.Linear(self.input_dim, start_ch * 4 * 4)

        blocks = []
        ch = start_ch
        spatial = 4
        self.attn_layers = nn.ModuleDict()
        for _ in range(num_upsamples):
            next_ch = max(base_channels, ch // 2)
            blocks.append(ResidualBlock(ch, next_ch, audio_embed_dim, label_nc=spade_label_nc, upsample=True))
            ch = next_ch
            spatial *= 2
            if spatial in self.attn_at:
                self.attn_layers[str(spatial)] = SelfAttention(ch)
        self.blocks = nn.ModuleList(blocks)

        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_channels, 3, 1, 1),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        def init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init)

    def forward(self, noise, audio_embedding):
        # use the audio MLP and normalize
        audio_embedding = F.normalize(self.audio_mlp(audio_embedding), dim=1)

        x = torch.cat([noise, audio_embedding], dim=1)
        x = self.fc(x).view(x.size(0), self.base_channels * 8, 4, 4)

        spatial = 4
        for blk in self.blocks:
            x = blk(x, audio_embedding)
            spatial *= 2
            key = str(spatial)
            if key in self.attn_layers:
                x = self.attn_layers[key](x)

        return self.to_rgb(x)

# -----------------------------
# Discriminator (Projection)
# -----------------------------

class Discriminator(nn.Module):
    """
    Projection discriminator (no BatchNorm, SpectralNorm on conv + linear).
    """
    def __init__(
        self,
        in_channels: int = 3,
        audio_embed_dim: int = 512,
        base_channels: int = 64,
        num_downsamples: int = 4
    ):
        super().__init__()
        c = base_channels
        layers = [
            spectral_norm(nn.Conv2d(in_channels, c, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)
        ]
        ch = c
        for _ in range(num_downsamples - 1):
            layers += [spectral_norm(nn.Conv2d(ch, ch * 2, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
            ch *= 2
            ch = min(ch, base_channels * 8)

        self.backbone = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_real = spectral_norm(nn.Linear(ch, 1))
        self.fc_proj = spectral_norm(nn.Linear(audio_embed_dim, ch))

        self._init_weights()

    def _init_weights(self):
        def init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init)

    # <<< added: expose pooled features for Feature Matching >>>
    def features(self, x):
        h = self.backbone(x)                # [B, C, H, W]
        h = self.gap(h).view(h.size(0), -1) # [B, C]
        return h

    def forward(self, x, audio_embedding):
        h = self.features(x)                # [B, C]
        real_score = self.fc_real(h)        # [B, 1]
        e = F.normalize(audio_embedding, dim=1)
        w = self.fc_proj(e)                 # [B, C]
        proj_score = (h * w).sum(dim=1, keepdim=True)
        return real_score + proj_score

# -----------------------------
# Builders
# -----------------------------

def build_models(
    resolution: int = 64,
    noise_dim: int = 128,
    audio_embed_dim: int = 512,
    base_channels: int = 64,
    spade_label_nc: int = 16,
):
    assert resolution in (64, 128, 256), "resolution must be 64, 128, or 256"
    num_ups = {64: 4, 128: 5, 256: 6}[resolution]
    num_down = num_ups

    G = Generator(
        noise_dim=noise_dim,
        audio_embed_dim=audio_embed_dim,
        out_channels=3,
        base_channels=base_channels,
        num_upsamples=num_ups,
        attn_at=(32, 64) if resolution >= 64 else (),
        spade_label_nc=spade_label_nc,
    )
    D = Discriminator(
        in_channels=3,
        audio_embed_dim=audio_embed_dim,
        base_channels=base_channels,
        num_downsamples=num_down,
    )
    return G, D
