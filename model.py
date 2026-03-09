import numpy as np
import torch
import torch.nn as nn
from config import *


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid = np.stack(np.meshgrid(
        np.arange(grid_size, dtype=np.float32),
        np.arange(grid_size, dtype=np.float32)
    ), axis=0).reshape(2, 1, grid_size, grid_size)

    half = embed_dim // 4
    def encode(pos):
        omega = 1.0 / (10000 ** (np.arange(half, dtype=np.float32) / half))
        out   = np.einsum('n,d->nd', pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb = np.concatenate([encode(grid[0]), encode(grid[1])], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), emb], axis=0)
    return emb


def patchify(images, patch_size=PATCH_SIZE):
    B, C, H, W = images.shape
    h, w = H // patch_size, W // patch_size
    x = images.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, patch_size * patch_size * C)
    return x


def unpatchify(patches, patch_size=PATCH_SIZE, img_size=IMG_SIZE):
    B, N, _ = patches.shape
    h = w = img_size // patch_size
    x = patches.reshape(B, h, w, patch_size, patch_size, 3)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, img_size, img_size)
    return x


def random_masking(x, mask_ratio):
    B, N, D = x.shape
    num_keep    = int(N * (1 - mask_ratio))
    noise       = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep    = ids_shuffle[:, :num_keep]
    x_visible   = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask        = torch.ones(B, N, device=x.device)
    mask[:, :num_keep] = 0
    mask        = torch.gather(mask, 1, ids_restore)
    return x_visible, mask, ids_restore


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3)
        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop),
        )
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3,
                 embed_dim=ENC_DIM, depth=ENC_LAYERS, num_heads=ENC_HEADS,
                 mlp_ratio=ENC_MLP, mask_ratio=MASK_RATIO, drop=DROP_RATE):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        G = img_size // patch_size

        self.patch_embed = nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))

        pe = get_2d_sincos_pos_embed(embed_dim, G, cls_token=True)
        self.register_buffer('pos_embed', torch.from_numpy(pe).float().unsqueeze(0))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, images):
        x = self.patch_embed(patchify(images, self.patch_size))
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = random_masking(x, self.mask_ratio)
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)
        for blk in self.blocks: x = blk(x)
        return self.norm(x), mask, ids_restore


class MAEDecoder(nn.Module):
    def __init__(self, num_patches=NUM_PATCHES, encoder_dim=ENC_DIM,
                 decoder_dim=DEC_DIM, depth=DEC_LAYERS, num_heads=DEC_HEADS,
                 mlp_ratio=DEC_MLP, patch_size=PATCH_SIZE, in_chans=3,
                 img_size=IMG_SIZE, drop=DROP_RATE):
        super().__init__()
        self.num_patches = num_patches
        self.patch_dim   = patch_size * patch_size * in_chans

        self.proj       = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        G  = img_size // patch_size
        pe = get_2d_sincos_pos_embed(decoder_dim, G, cls_token=True)
        self.register_buffer('pos_embed', torch.from_numpy(pe).float().unsqueeze(0))

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads, mlp_ratio, drop=drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, self.patch_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, latent, ids_restore):
        B     = latent.size(0)
        x     = self.proj(latent)
        N_vis = x.size(1) - 1
        N_msk = self.num_patches - N_vis

        x_full = torch.cat([x[:, 1:, :],
                            self.mask_token.expand(B, N_msk, -1)], dim=1)
        x_full = torch.gather(x_full, 1,
                    ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))

        x = torch.cat([x[:, :1, :], x_full], dim=1) + self.pos_embed
        for blk in self.blocks: x = blk(x)
        return self.pred(self.norm(x)[:, 1:, :])


class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MAEEncoder()
        self.decoder = MAEDecoder()

    def forward(self, images, mask_ratio=MASK_RATIO):
        self.encoder.mask_ratio = mask_ratio
        latent, mask, ids_restore = self.encoder(images)
        pred = self.decoder(latent, ids_restore)
        return pred, mask