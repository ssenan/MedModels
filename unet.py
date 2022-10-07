import enum
import math
from functools import partial
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    
    return d() if isfunction(d) else d

class TimeEmbedding(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        freqs = torch.exp(
            torch.arange(0, self.dim, 2, dtype=torch.float32) * 
            (-math.log(10000) / self.dim)).to(device=input.device)
        args = input[:, None] * freqs[None, :]
        embedding = torch.cat((args.sin(), args.cos()), dim=-1)
        return embedding

#--------------------------------------
class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

# Blocks for Unet

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        #self.block = nn.Sequential(
            #nn.Conv3d(dim, dim_out, 3, padding=1),
            #nn.GroupNorm(groups, dim_out),
            #nn.SiLU(),
        #)
        self.conv = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.silu = nn.SiLU()

    def forward(self, x, scale_shift=None):
        #return self.block(x)
        x = self.conv(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x *(scale+1) + shift

        x = self.silu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
            ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
    def forward(self, x, time_emb=None):
        
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()

        self.num_heads=num_heads
        self.norm = nn.GroupNorm(4, in_channels)
        self.qkv = nn.Conv3d(in_channels, in_channels * 3, 1, bias=False) 
        self.out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        batch, channel, depth,  height, width = x.shape
        norm = self.norm(x)
        qkv = self.qkv(norm)
        q, k, v = rearrange(qkv, 'b (qkv heads c) d h w -> qkv b heads c (d h w)', heads=self.num_heads, qkv=3)
        k = k.softmax(dim=-1)
        sim = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', sim, q)
        out = rearrange(out, 'b heads c (d h w) -> b (heads c) d h w', heads=self.num_heads, h=height, w=width)
        out = self.out(out)
        return out + x
        


class Unet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        model_channels = 32,
        channel_mults = (1, 2, 4, 8),
        num_res_blocks=3,
        attention_resolutions=[8],
        use_time_emb = True,
        res_blocks = 8,
        dropout = 0,
    ):

        super().__init__()
        
        self.initConv = nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1)
        dims = [model_channels, *map(lambda m: model_channels * m, channel_mults)]
        in_out = list(zip(dims[:1], dims[1:]))
        
        
        # Time embedding
        if use_time_emb:
            time_dim = model_channels
            self.time_mlp = nn.Sequential(
                TimeEmbedding(model_channels),
                nn.Linear(model_channels, model_channels*4),
                nn.SiLU(),
                nn.Linear(model_channels*4, model_channels)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.down = nn.ModuleList([])


        for i, (in_channel, out_channel) in enumerate(in_out):
            is_last = (i >= len(in_out) - 1)

            self.down.append(nn.ModuleList([
                ResBlock(in_channel, out_channel, time_emb_dim=time_dim),
                ResBlock(out_channel, out_channel, time_emb_dim=time_dim),
                AttentionBlock(out_channel),
                Downsample(out_channel) if not is_last 
                                                    else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.midBlock1 = ResBlock(model_channels, model_channels, time_emb_dim=time_dim) 
        self.midAttn = AttentionBlock(model_channels)
        self.midBlock2 = ResBlock(model_channels, model_channels, time_emb_dim=time_dim)

        self.up = nn.ModuleList([])
        for i, (in_channel, out_channel) in enumerate(reversed(in_out[1:])):
            is_last = (i >= len(in_out) - 1)

            self.up.append(nn.ModuleList([
                ResBlock(out_channel + in_channel , in_channel, time_emb_dim=time_dim),
                ResBlock(in_channel, in_channel, time_emb_dim=time_dim),
                AttentionBlock(in_channel),
                Upsample(in_channel) if not is_last 
                                                else nn.Identity()
            ]))
        
        self.outConv = nn.Conv3d(model_channels, out_channels, 1)


    def forward(self, x, time):
        
        t = self.time_mlp(time)
        x = self.initConv(x)
        r = x.clone()
        h = []

        for block1, block2, attn, downsample in self.down:
            x = block1(x,t)
            h.append(x)

            x = block2(x,t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.midBlock1(x,t)
        x = self.midAttn(x)
        x = self.midBlock2(x,t)

        for block1, block2, attn, upsample in self.up:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x,t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x,t)
            x = attn(x)

            x = upsample(x)
        #x = torch.cat((x,r), dim=1)
        return self.outConv(x)



        
