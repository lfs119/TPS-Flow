import math
from typing import *

import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_temporal_window(x, t, method='mix'):
        # " Create temporal window for the input tensor based on the specified method."
        assert method in ['roll', 'prv', 'first', 'mix']

        if method == 'roll':
            # m = einops.rearrange(x, '(b t) d c -> b t d c', t=t)
            m = x
            l = torch.roll(m, shifts=1, dims=1)
            r = torch.roll(m, shifts=-1, dims=1)

            recon = torch.cat([l, m, r], dim=-1)
            del l, m, r

            # recon = einops.rearrange(recon, 'b t d c -> (b t) d c')
            return recon

        if method == 'prv':
            # x = einops.rearrange(x, '(b t) d c -> b t d c', t=t)
            prv = torch.cat([x[:, :1], x[:, :-1]], dim=1)

            recon = torch.cat([x, prv], dim=-1)
            del x, prv

            # recon = einops.rearrange(recon, 'b t d c -> (b t) d c')
            return recon

        if method == 'first':
            # x = einops.rearrange(x, '(b t) d c -> b t d c', t=t)
            prv = x[:, [0], :, :].repeat(1, t, 1, 1)

            recon = torch.cat([x, prv], dim=-1)
            del x, prv

            # recon = einops.rearrange(recon, 'b t d c -> (b t) d c')
            return recon
        
        if method == 'mix':
            # x = einops.rearrange(x, '(b t) d c -> b t d c', t=t)
            first_prv = x[:, [0], :, :].repeat(1, t, 1, 1)
            last_prv = x[:, [-1], :, :].repeat(1, t, 1, 1)

            recon = torch.cat([x, first_prv, last_prv], dim=-1)
            del x, first_prv, last_prv

            # recon = einops.rearrange(recon, 'b t d c -> (b t) d c')
            return recon
        
class Temporal_Windowing(nn.Module):
    def __init__(self, dim, method='mix'):
        super().__init__()
        self.method = method
        self.embed_dim = dim
        self.temporal_window_dense = nn.Linear(3 * self.embed_dim, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim * 3, eps=1e-12)
        # self.norm1 = nn.LayerNorm(self.embed_dim, eps=1e-12)
        self.dense1_act = nn.GELU()

    def forward(self, x, T):
        temporal_w_out = make_temporal_window(x, T, self.method)
        temporal_w_out_enocder = self.temporal_window_dense(self.norm(temporal_w_out))
        x = self.dense1_act(temporal_w_out_enocder)
        return x

# from  https://github.com/jiachenzhu/DyT      
class DyT(nn.Module):
    def __init__(self, num_features, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1)*alpha_init)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class TPA(nn.Module):
    def __init__(self, d_model, n_heads, rank_ratio=0.25, seq_len=999):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rank = max(n_heads, int(d_model * rank_ratio))
        assert self.rank % n_heads == 0, f"rank({self.rank}) must be divisible by n_heads({n_heads})"
        self.rank_per_head = self.rank // n_heads

        self.Wq = nn.Linear(d_model, self.rank, bias=False)
        self.Wk = nn.Linear(d_model, self.rank, bias=False)
        self.Wv = nn.Linear(d_model, self.rank, bias=False)

        self.U = nn.Parameter(torch.Tensor(n_heads, self.head_dim, self.rank_per_head))
        self.V = nn.Parameter(torch.Tensor(n_heads, self.rank_per_head, self.head_dim))

        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

        # RoPE
        self.seq_len = seq_len
        self.register_buffer('freqs', self._precompute_freqs())

    def _precompute_freqs(self):
        theta = 1.0 / (10000 ** (torch.arange(0, self.head_dim // 2, 1).float() / (self.head_dim // 2)))
        seq_len = self.seq_len
        t = torch.arange(seq_len, device=theta.device)
        freqs = torch.outer(t, theta)
        return torch.polar(torch.ones_like(freqs), freqs)

    def _apply_rope(self, x):
        B, L, H, D = x.shape  # [batch, seq_len, heads, head_dim]
        x_complex = torch.view_as_complex(x.reshape(B, L, H, D // 2, 2).contiguous())
        rotated = x_complex * self.freqs[:L].unsqueeze(0).unsqueeze(2)
        return torch.view_as_real(rotated).reshape(B, L, H, D)

    def forward(self, x, mask):
        # 输入维度: (batch, L, d_model)
        B, L, _ = x.shape

        q_low = self.Wq(x).view(B, L, self.n_heads, self.rank_per_head)
        k_low = self.Wk(x).view(B, L, self.n_heads, self.rank_per_head)
        v_low = self.Wv(x).view(B, L, self.n_heads, self.rank_per_head)

        q = torch.einsum('hdr,blhr->blhd', self.U, q_low)  # [B,L,H,D]
        k = torch.einsum('hdr,blhr->blhd', self.U, k_low)

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        mask = mask.type(torch.bool)
        attn = torch.einsum('blhd,bmhd->bhlm', q, k) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        output = torch.einsum('bhlm,bmhr->blhr', attn, v_low)
        output = torch.einsum('blhr,hrd->blhd', output, self.V)

        return output.reshape(B, L, self.d_model)
    

class ProteinSpatioTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, time_heads=8):
        super().__init__()
        self.aa_attn = TPA(dim, num_heads)
        self.time_attn = TPA(dim, time_heads)
        self.proj = nn.Linear(dim, dim)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-12) 
        self.layer_norm3 = nn.LayerNorm(dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1) 
        self.gate_time = nn.Sequential( 
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.gate_aa = nn.Sequential( 
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # self.gate_proj = nn.Sequential( 
        #     nn.Linear(dim * 2, dim),
        #     nn.Sigmoid()
        # )
             
        
    def forward(self, x, mask, pe, xr=None, xf=None):
        """输入形状: (B, T, L, C)"""
        B, T, L, C = x.shape
        residual_time = x
        x = self.layer_norm1(x)
        x_time_mask = einops.rearrange(mask, 'B T L -> (B L) T')
        x_time = einops.rearrange(x, 'B T L C -> (B L) T C')  
        x_time = self.time_attn(x_time, x_time_mask)
        x_time = einops.rearrange(x_time, '(B L) T C -> B T L C', B=B, L=L)
        # x_time = residual_time + x_time

        gate = self.gate_time(torch.cat([residual_time, x_time], dim=-1))
        x_time = gate * residual_time + (1 - gate) * x_time
        
        residual_aa = x_time
        # 0417
        x_time = self.layer_norm2(x_time)
        # x_time = self.layer_norm2(x_time) + pe
        x_aa_mask = einops.rearrange(mask, 'B T L -> (B T) L')
        x_aa = einops.rearrange(x_time, 'B T L C -> (B T) L C')  
        x_aa = self.aa_attn(x_aa, x_aa_mask)
        x_aa = einops.rearrange(x_aa, '(B T) L C -> B T L C', B=B, T=T)
        # x_aa = residual_aa + x_aa

        gate = self.gate_aa(torch.cat([residual_aa, x_aa], dim=-1))
        x_aa = gate * residual_aa + (1 - gate) * x_aa

        residual_proj = x_aa
        x_aa = self.dropout(self.proj(self.layer_norm3(x_aa)))
        output = x_aa + residual_proj

        # gate = self.gate_proj(torch.cat([residual_proj, x_aa], dim=-1))
        # output = gate * residual_proj + (1 - gate) * x_aa

                
        return output, None
    
class ProteinSpatioTemporalAttention2(nn.Module):
    def __init__(self, dim, num_heads=8, time_heads=8):
        super().__init__()
        self.aa_attn = TPA(dim, num_heads)
        self.time_attn = TPA(dim, time_heads)
        self.layer_norm1 = DyT(dim)
        self.layer_norm2 = DyT(dim) 
        self.layer_norm3 = DyT(dim)
        self.gate_time = nn.Sequential( 
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.gate_aa = nn.Sequential( 
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.gate_proj = nn.Sequential( 
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
       
             
        
    def forward(self, x, mask, t, xr=None, xf=None):
        """输入形状: (B, T, L, C)"""
        B, T, L, C = x.shape
        residual_time = x
        x = self.layer_norm1(x)
        x_time_mask = einops.rearrange(mask, 'B T L -> (B L) T')
        x_time = einops.rearrange(x, 'B T L C -> (B L) T C')  
        x_time = self.time_attn(x_time, x_time_mask)
        x_time = einops.rearrange(x_time, '(B L) T C -> B T L C', B=B, L=L)
        # x_time = residual_time + x_time

        gate = self.gate_time(t).unsqueeze(1)
        x_time = gate * residual_time + (1 - gate) * x_time
        
        residual_aa = x_time
        x_time = self.layer_norm2(x_time)
        x_aa_mask = einops.rearrange(mask, 'B T L -> (B T) L')
        x_aa = einops.rearrange(x_time, 'B T L C -> (B T) L C')  
        x_aa = self.aa_attn(x_aa, x_aa_mask)
        x_aa = einops.rearrange(x_aa, '(B T) L C -> B T L C', B=B, T=T)
        # x_aa = residual_aa + x_aa

        gate = self.gate_aa(t).unsqueeze(1)
        x_aa = gate * residual_aa + (1 - gate) * x_aa
        

        moe_loss = 0.0
        residual_proj = x_aa
        proj_out = self.proj2(self.act(self.proj1(self.layer_norm3(x_aa))))
        gate = self.gate_proj(t).unsqueeze(1)
        output = gate * residual_proj + (1 - gate) * proj_out

                
        return output, moe_loss
    
class ProteinSpatioTemporalAttention3(nn.Module):
    def __init__(self, dim, num_heads=8, time_heads=8):
        super().__init__()
        self.aa_attn = TPA(dim, num_heads)
        self.time_attn = TPA(dim, time_heads)
        self.layer_norm1 = DyT(dim)
        self.layer_norm2 = DyT(dim) 
        self.layer_norm3 = DyT(dim)
        self.gate_time = nn.Sequential( 
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.gate_aa = nn.Sequential( 
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        # self.gate_proj = nn.Sequential( 
        #     nn.Linear(dim, dim),
        #     nn.Sigmoid()
        # )
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
       
             
        
    def forward(self, x, mask, t, xr=None, xf=None):
        """输入形状: (B, T, L, C)"""
        B, T, L, C = x.shape
        residual_time = x
        x = self.layer_norm1(x)
        x_time_mask = einops.rearrange(mask, 'B T L -> (B L) T')
        x_time = einops.rearrange(x, 'B T L C -> (B L) T C')  
        x_time = self.time_attn(x_time, x_time_mask)
        x_time = einops.rearrange(x_time, '(B L) T C -> B T L C', B=B, L=L)
        # x_time = residual_time + x_time

        # gate = self.gate_time(torch.cat([residual_time, x_time], dim=-1))
        # x_time = gate * residual_time + (1 - gate) * x_time

        gate = self.gate_time(t).unsqueeze(1)
        x_time = gate * residual_time + (1 - gate) * x_time
        
        residual_aa = x_time
        x_time = self.layer_norm2(x_time)
        x_aa_mask = einops.rearrange(mask, 'B T L -> (B T) L')
        x_aa = einops.rearrange(x_time, 'B T L C -> (B T) L C')  
        x_aa = self.aa_attn(x_aa, x_aa_mask)
        x_aa = einops.rearrange(x_aa, '(B T) L C -> B T L C', B=B, T=T)
        # x_aa = residual_aa + x_aa

        # gate = self.gate_aa(torch.cat([residual_aa, x_aa], dim=-1))
        # x_aa = gate * residual_aa + (1 - gate) * x_aa

        gate = self.gate_aa(t).unsqueeze(1)
        x_aa = gate * residual_aa + (1 - gate) * x_aa
        

        moe_loss = 0.0
        residual_proj = x_aa
        proj_out = self.proj2(self.act(self.proj1(self.layer_norm3(x_aa))))
        output = residual_proj + proj_out
        # gate = self.gate_proj(t).unsqueeze(1)
        # output = gate * residual_proj + (1 - gate) * proj_out

                
        return output, moe_loss
    
class ProteinSpatioTemporalAttention5(nn.Module):
    def __init__(self, dim, num_heads=8, time_heads=8):
        super().__init__()
        self.aa_attn = TPA(dim, num_heads)
        self.time_attn = TPA(dim, time_heads)
        self.proj = nn.Linear(dim, dim)
        self.layer_norm1 = DyT(dim)
        self.layer_norm2 = DyT(dim) 
        self.layer_norm3 = DyT(dim)
        self.dropout = nn.Dropout(0.1) 
        self.gate_time = nn.Sequential( 
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.gate_aa = nn.Sequential( 
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # self.gate_proj = nn.Sequential( 
        #     nn.Linear(dim * 2, dim),
        #     nn.Sigmoid()
        # )
             
        
    def forward(self, x, mask, t, xr=None, xf=None):
        """输入形状: (B, T, L, C)"""
        B, T, L, C = x.shape
        residual_time = x
        x = self.layer_norm1(x)
        x_time_mask = einops.rearrange(mask, 'B T L -> (B L) T')
        x_time = einops.rearrange(x, 'B T L C -> (B L) T C')  
        x_time = self.time_attn(x_time, x_time_mask)
        x_time = einops.rearrange(x_time, '(B L) T C -> B T L C', B=B, L=L)

        gate = self.gate_time(torch.cat([residual_time, x_time], dim=-1))
        x_time = gate * residual_time + (1 - gate) * x_time
        
        residual_aa = x_time
        # 0417
        x_time = self.layer_norm2(x_time)
        # x_time = self.layer_norm2(x_time) + pe
        x_aa_mask = einops.rearrange(mask, 'B T L -> (B T) L')
        x_aa = einops.rearrange(x_time, 'B T L C -> (B T) L C')  
        x_aa = self.aa_attn(x_aa, x_aa_mask)
        x_aa = einops.rearrange(x_aa, '(B T) L C -> B T L C', B=B, T=T)
        # x_aa = residual_aa + x_aa

        gate = self.gate_aa(torch.cat([residual_aa, x_aa], dim=-1))
        x_aa = gate * residual_aa + (1 - gate) * x_aa

        residual_proj = x_aa
        x_aa = self.dropout(self.proj(self.layer_norm3(x_aa)))
        output = x_aa + residual_proj

        # gate = self.gate_proj(torch.cat([residual_proj, x_aa], dim=-1))
        # output = gate * residual_proj + (1 - gate) * x_aa

                
        return output, None
    
class ProteinSpatioTemporalAttention6(nn.Module):
    def __init__(self, dim, num_heads=12, time_heads=12):
        super().__init__()
        self.aa_attn = TPA(dim, num_heads)
        self.time_attn = TPA(dim, time_heads)
        self.proj = nn.Linear(dim, dim)
        self.layer_norm1 = DyT(dim)
        self.layer_norm2 = DyT(dim) 
        self.layer_norm3 = DyT(dim)
        self.dropout = nn.Dropout(0.1) 
        self.gate_time = nn.Sequential( 
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.gate_aa = nn.Sequential( 
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # self.gate_proj = nn.Sequential( 
        #     nn.Linear(dim * 2, dim),
        #     nn.Sigmoid()
        # )
             
        
    def forward(self, x, mask, t, xr=None, xf=None):
        """输入形状: (B, T, L, C)"""
        B, T, L, C = x.shape
        residual_time = x
        x = self.layer_norm1(x)
        x_time_mask = einops.rearrange(mask, 'B T L -> (B L) T')
        x_time = einops.rearrange(x, 'B T L C -> (B L) T C')  
        x_time = self.time_attn(x_time, x_time_mask)
        x_time = einops.rearrange(x_time, '(B L) T C -> B T L C', B=B, L=L)

        gate = self.gate_time(torch.cat([residual_time, x_time], dim=-1))
        x_time = gate * residual_time + (1 - gate) * x_time
        
        residual_aa = x_time
        # 0417
        x_time = self.layer_norm2(x_time)
        # x_time = self.layer_norm2(x_time) + pe
        x_aa_mask = einops.rearrange(mask, 'B T L -> (B T) L')
        x_aa = einops.rearrange(x_time, 'B T L C -> (B T) L C')  
        x_aa = self.aa_attn(x_aa, x_aa_mask)
        x_aa = einops.rearrange(x_aa, '(B T) L C -> B T L C', B=B, T=T)
        # x_aa = residual_aa + x_aa

        gate = self.gate_aa(torch.cat([residual_aa, x_aa], dim=-1))
        x_aa = gate * residual_aa + (1 - gate) * x_aa

        residual_proj = x_aa
        x_aa = self.dropout(self.proj(self.layer_norm3(x_aa)))
        output = x_aa + residual_proj

        # gate = self.gate_proj(torch.cat([residual_proj, x_aa], dim=-1))
        # output = gate * residual_proj + (1 - gate) * x_aa

                
        return output, None
    
class ProteinSpatioTemporalAttention7(nn.Module):
    def __init__(self, dim, num_heads=8, time_heads=8):
        super().__init__()
        self.aa_attn = TPA(dim, num_heads)
        self.time_attn = TPA(dim, time_heads)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-12) 
        self.layer_norm3 = nn.LayerNorm(dim, eps=1e-12)
        self.gate_time = nn.Sequential( 
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )
        self.gate_aa = nn.Sequential( 
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )
        
        self.proj1 = nn.Linear(dim, dim*4)
        self.proj2 = nn.Linear(dim*4, dim)
        self.act = nn.GELU()
       
             
        
    def forward(self, x, mask, t, xr=None, xf=None):
        """输入形状: (B, T, L, C)"""
        B, T, L, C = x.shape
        residual_time = x
        x = self.layer_norm1(x)
        x_time_mask = einops.rearrange(mask, 'B T L -> (B L) T')
        x_time = einops.rearrange(x, 'B T L C -> (B L) T C')  
        x_time = self.time_attn(x_time, x_time_mask)
        x_time = einops.rearrange(x_time, '(B L) T C -> B T L C', B=B, L=L)
        # x_time = residual_time + x_time

        gate = self.gate_time(torch.cat([residual_time, x_time], dim=-1))
        x_time = gate * residual_time + (1 - gate) * x_time
        
        residual_aa = x_time
        x_time = self.layer_norm2(x_time)
        x_aa_mask = einops.rearrange(mask, 'B T L -> (B T) L')
        x_aa = einops.rearrange(x_time, 'B T L C -> (B T) L C')  
        x_aa = self.aa_attn(x_aa, x_aa_mask)
        x_aa = einops.rearrange(x_aa, '(B T) L C -> B T L C', B=B, T=T)
        # x_aa = residual_aa + x_aa

        gate = self.gate_aa(torch.cat([residual_aa, x_aa], dim=-1))
        x_aa = gate * residual_aa + (1 - gate) * x_aa

        # gate = self.gate_aa(t).unsqueeze(1)
        # x_aa = gate * residual_aa + (1 - gate) * x_aa
        

        moe_loss = 0.0
        residual_proj = x_aa
        proj_out = self.proj2(self.act(self.proj1(self.layer_norm3(x_aa))))
        output = residual_proj + proj_out

                
        return output, moe_loss
    

class ProteinSpatioTemporalAttention8(nn.Module):
    def __init__(self, dim, num_heads=8, time_heads=8):
        super().__init__()
        self.aa_attn = TPA(dim, num_heads)
        self.time_attn = TPA(dim, time_heads)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-12) 
        self.layer_norm3 = nn.LayerNorm(dim, eps=1e-12)
        self.gate_time = nn.Sequential( 
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )
        self.gate_aa = nn.Sequential( 
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )
        
        self.proj1 = nn.Linear(dim, dim*4)
        self.proj2 = nn.Linear(dim*4, dim)
        self.act = nn.GELU()

        self.tem_win = Temporal_Windowing(dim)
       
             
        
    def forward(self, x, mask, t, xr=None, xf=None):
        """输入形状: (B, T, L, C)"""
        B, T, L, C = x.shape
        residual_time = x
        x = self.layer_norm1(x)
        x_time_mask = einops.rearrange(mask, 'B T L -> (B L) T')
        x_time = einops.rearrange(x, 'B T L C -> (B L) T C')  
        x_time = self.time_attn(x_time, x_time_mask)
        x_time = einops.rearrange(x_time, '(B L) T C -> B T L C', B=B, L=L)
        # x_time = residual_time + x_time

        gate = self.gate_time(torch.cat([residual_time, x_time], dim=-1))
        x_time = gate * residual_time + (1 - gate) * x_time
        
        residual_aa = x_time
        x_time = self.layer_norm2(x_time)
        x_aa_mask = einops.rearrange(mask, 'B T L -> (B T) L')
        x_aa = einops.rearrange(x_time, 'B T L C -> (B T) L C')  
        x_aa = self.aa_attn(x_aa, x_aa_mask)
        x_aa = einops.rearrange(x_aa, '(B T) L C -> B T L C', B=B, T=T)
        # x_aa = residual_aa + x_aa

        gate = self.gate_aa(torch.cat([residual_aa, x_aa], dim=-1))
        x_aa = gate * residual_aa + (1 - gate) * x_aa

        # gate = self.gate_aa(t).unsqueeze(1)
        # x_aa = gate * residual_aa + (1 - gate) * x_aa
        

        moe_loss = 0.0
        residual_proj = x_aa
        proj_out = self.proj2(self.act(self.proj1(self.layer_norm3(x_aa))))
        output = residual_proj + proj_out

        output = self.tem_win(output, T)

                
        return output, moe_loss
    
class FreBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_dim=1):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels)
        )
        self.processpha = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels)
        )
        self.gate = nn.Sequential( 
            nn.Linear(in_channels * 2, in_channels),
            nn.Sigmoid()
        )
        self.apply_dim = apply_dim

    def forward(self, x):
        xori = x
        x_freq = torch.fft.fft(x, dim=self.apply_dim if hasattr(self, 'apply_dim') else 1, norm='ortho')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out1 = torch.fft.ifft(x_out, dim=self.apply_dim if hasattr(self, 'apply_dim') else 1, norm='ortho').real 
        gate = self.gate(torch.cat([xori, x_out1], dim=-1))
        return gate * xori + (1 - gate) * x_out1

class MultiScaleProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 输入维度根据实际特征设计
        self.proj_coarse = nn.Linear(3, dim)  # 平移特征
        self.proj_medium = nn.Linear(4, dim)  # 旋转特征
        self.proj_fine = nn.Linear(3, dim)    # 交互特征
    
    def forward(self, x):
        # x : [B, N, 7] (4D + 3D)
        rotation = x[..., :4]
        translation = x[..., 4:]
        
        # 细粒度交互: 旋转虚部 * 平移
        interaction = rotation[..., 1:] * translation  # [B, N, 3]
        
        # 多尺度投影
        coarse = self.proj_coarse(translation)  # 粗粒度: 平移
        medium = self.proj_medium(rotation)     # 中粒度: 旋转
        fine = self.proj_fine(interaction)      # 细粒度: 旋转-平移交互
        
        return torch.cat([coarse, medium, fine], dim=-1)  # [B, N, 384]
    
class FreBlock2(nn.Module):
    def __init__(self, in_channels, h=1, w=1):
        super(FreBlock2, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, in_channels, 2, dtype=torch.float32) * 0.02)
        self.dim = in_channels
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-12)
        self.gate = nn.Sequential( 
            nn.Linear(in_channels * 2, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x, spatial_size=None):
        assert x.ndim==4, "Input tensor must have 4 dimensions (B, T, L, C)"
        B, T,  L, C = x.shape
        xori = x
        x = self.norm1(x)
        if spatial_size is None:
            a = T 
            b = L  
        else:
            a, b = spatial_size
            assert a * b == L, "spatial_size must match sequence length L"
        x = x.view(B, a, b, C).to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight) #(h,w,c)
        x = x * weight.unsqueeze(0)  

        x_out1 = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x_out1 = x_out1.view(B, T, L, C) 
        gate = self.gate(torch.cat([xori, x_out1], dim=-1))
        return gate * xori + (1 - gate) * x_out1
        
    
class FreBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels)
        )
        self.processpha = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels)
        )
        self.gate = nn.Sequential( 
            nn.Linear(in_channels * 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        xori = x
        x_freq = torch.fft.fft(x, dim=1, norm='ortho')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out1 = torch.fft.ifft(x_out, dim=1, norm='ortho').real 
        gate = self.gate(torch.cat([xori, x_out1], dim=-1))
        return gate * xori + (1 - gate) * x_out1

