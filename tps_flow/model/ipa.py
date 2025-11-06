# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .primitives import Linear, ipa_point_weights_init_

"""
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
"""
from ..rigid_utils import Rotation, Rigid
from ..tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


class InvariantPointAttention(nn.Module):
    def __init__(self, c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points, inf=1e5, eps=1e-8, dropout=0.0):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        self.dropout = dropout
        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        hpv = self.no_heads * self.no_v_points * 3

        if self.c_z > 0:
            self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(self, s, r, z=None, frame_mask=None, attn_mask=None):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        
        z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3] # 4 * 32 * 3 - > 4 * 32
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]   （4，8，3）
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        if self.c_z > 0:
            b = self.linear_b(z[0])

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )

        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        if self.c_z:
            a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att**2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        
        if frame_mask is not None:
            square_mask = frame_mask.unsqueeze(-1) * frame_mask.unsqueeze(-2)
            square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

  
        a = a + pt_att
        if frame_mask is not None:
            a = a + square_mask.unsqueeze(-3)
        if attn_mask is not None:
            attn_mask = self.inf * (attn_mask - 1)
            a = a + attn_mask

        a = self.softmax(a)
        a = F.dropout(a, p=self.dropout, training=self.training)
        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
   
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, H, C_z]
        if self.c_z > 0:
            o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

            # [*, N_res, H * C_z]
            o_pair = flatten_final_dims(o_pair, 2)

            # [*, N_res, C_s]
            s = self.linear_out(
                torch.cat(
                    (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
                ).to(dtype=z[0].dtype)
            )
        else:
            s = self.linear_out(
                torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1).to(
                    dtype=s.dtype
                )
            )
        return s
    

class EnhancedIPA(nn.Module):
    def __init__(self, c_s: int, c_z: int, c_hidden: int, 
                 no_heads: int, no_qk_points: int, no_v_points: int,
                 inf: float = 1e5, eps: float = 1e-8, dropout: float = 0.0):
        super().__init__()
        
        # 基础参数
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        self.dropout = dropout
        
        hc = self.c_hidden * self.no_heads
        # 标量注意力投影
        self.linear_q = nn.Linear(c_s, c_hidden * no_heads)
        self.linear_kv = nn.Linear(c_s, 2 * c_hidden * no_heads)

        # 几何点生成（坐标+方向）
        self.linear_q_points = nn.Linear(c_s, 6 * no_heads * no_qk_points)
        self.linear_kv_points = nn.Linear(c_s, 6 * no_heads * (no_qk_points + no_v_points))
        

        # 动态几何门控
        self.point_gate = nn.Linear(c_s, no_heads * no_qk_points)

        # 几何注意力参数
        self.geom_weights = nn.Parameter(torch.ones(3))  # 距离/方向/曲率权重
        self.dir_weight = nn.Parameter(torch.tensor(0.5))
        self.head_weights = nn.Parameter(torch.zeros(no_heads))

        # 输出层
        concat_dim = no_heads * (c_hidden + 7 * no_v_points)  # 3+3+1
        if c_z > 0:
            concat_dim += no_heads * c_z
        self.linear_out = nn.Linear(concat_dim, c_s)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        nn.init.normal_(self.head_weights, mean=0.0, std=0.02)
        nn.init.constant_(self.dir_weight, 0.5)
        nn.init.constant_(self.geom_weights, 1.0)
    
    def forward(self, 
               s: torch.Tensor, 
               r,
               z: Optional[torch.Tensor] = None,
               frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        #######################################
        # 1. 特征投影
        #######################################
        # 标量投影
        q = self.linear_q(s)  # [..., H*C]
        kv = self.linear_kv(s)  # [..., 2*H*C]
        split_size = self.c_hidden * self.no_heads
        k, v = torch.split(kv, [split_size, split_size], dim=-1)
        
        # 几何点生成
        q_pts = self.linear_q_points(s)  # [..., H*P_q*6]
        kv_pts = self.linear_kv_points(s)
        k_pts, v_pts = torch.split(kv_pts, 
                                 [self.no_qk_points*6*self.no_heads, self.no_v_points*6*self.no_heads], 
                                 dim=-1)
        
        # 重塑维度
        q = rearrange(q, '... (h c) -> ... h c', h=self.no_heads, c=self.c_hidden)
        k = rearrange(k, '... (h c) -> ... h c', h=self.no_heads, c=self.c_hidden)
        v = rearrange(v, '... (h c) -> ... h c', h=self.no_heads, c=self.c_hidden)
        
        q_pts = self._process_points(q_pts, self.no_qk_points, r)
        k_pts = self._process_points(k_pts, self.no_qk_points, r)
        v_pts = self._process_points(v_pts, self.no_v_points, r)

        #######################################
        # 3. 动态几何门控
        #######################################
        gate = torch.sigmoid(self.point_gate(s))
        gate = rearrange(gate, '... (h p) -> ... h p 1', h=self.no_heads, p=self.no_qk_points)
        q_pts = q_pts * gate

        #######################################
        # 4. 注意力计算
        #######################################
        # 标量注意力
        scalar_att = torch.einsum('bqhc,bkhc->bhqk', q, k) / math.sqrt(self.c_hidden)
        
        # 几何注意力
        k_pts, v_pts = k_pts.unsqueeze(0), v_pts.unsqueeze(0)
        pos_att = self._position_attention(q_pts[..., :3], k_pts[..., :3])
        dir_att = self._direction_attention(q_pts[..., 3:], k_pts[..., 3:])
        geom_att = (self.geom_weights[0] * pos_att + 
                   self.geom_weights[1] * dir_att)
        
        # 合并注意力
        att = scalar_att + self.dir_weight * geom_att
        att = att * self.head_weights.view(-1, 1, 1)  # [B, H, L, L]
        
        # 掩码处理
        if frame_mask is not None:
            mask_2d = frame_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            mask_2d = mask_2d * mask_2d.transpose(-1, -2)    # [B, 1, L, L]
            att += self.inf * (mask_2d - 1)    # [B, H, L, L]
        
        att = torch.softmax(att, dim=-1)

        #######################################
        # 5. 特征聚合
        #######################################
        o_scalar = torch.einsum('bhqk,bkhc->bqhc', att, v)
        o_geom = torch.einsum('bhqk,bkhpd->bqhpd', att, v_pts)
        o_geom = self._process_output_geometry(o_geom, r)

        output = torch.cat([
            o_scalar.flatten(-2), 
            o_geom.flatten(-3,-1)
        ], dim=-1)
        
        if self.c_z > 0:
            o_pair = torch.einsum('...hij,...ijk->...ihk', att, z)
            output = torch.cat([output, o_pair.flatten(-2)], dim=-1)

        return self.linear_out(output)

    def _process_points(self, pts: torch.Tensor, num_points: int, r):
        pts = rearrange(pts, '... (h p d) -> ... h p d', h=self.no_heads, p=num_points, d=6)
        coords, dirs = pts[..., :3], pts[..., 3:]
        coords = r[...,None,None].apply(coords).reshape(-1, self.no_heads, num_points, 3)
        dirs = r[...,None,None].get_rots().apply(dirs).reshape(-1, self.no_heads, num_points, 3)
        return torch.cat([coords, dirs], dim=-1)

    def _process_output_geometry(self, geom: torch.Tensor, rigid: 'Rigid') -> torch.Tensor:
        global_coords = geom[..., :3]
        global_dirs = geom[..., 3:6]

        inv_rot = rigid.get_rots().invert()
        trans_inv = -inv_rot.apply(rigid.get_trans())

        local_coords = inv_rot[..., None, None].apply(global_coords) + trans_inv.unsqueeze(-2).unsqueeze(-2)
        local_dirs = inv_rot[..., None, None].apply(global_dirs)
        local_dirs = F.normalize(local_dirs, dim=-1)

        local_norms = torch.norm(local_coords, dim=-1, keepdim=True)

        return torch.cat([local_coords, local_dirs, local_norms], dim=-1)

    def _position_attention(self, q_coords: torch.Tensor, k_coords: torch.Tensor) -> torch.Tensor:
        B, L_q, H, Pq, _ = q_coords.shape
        L_k = k_coords.shape[1]
        
        q_exp = q_coords.unsqueeze(2)
        k_exp = k_coords.unsqueeze(1)
        delta = q_exp - k_exp
        
        linear = torch.norm(delta, dim=-1)
        log = torch.log(linear + self.eps)
        inv = 1 / (linear + self.eps)
        
        combined = (linear + log + inv).mean(dim=(-1))  # [B, L_q, L_k, H]
        
        return combined.permute(0, 3, 1, 2)  # [B, H, L_q, L_k]

    def _direction_attention(self, q_dirs: torch.Tensor, k_dirs: torch.Tensor) -> torch.Tensor:
        B, L_q, H, Pq, _ = q_dirs.shape
        L_k = k_dirs.shape[1]

        q_exp = q_dirs.unsqueeze(2)
        k_exp = k_dirs.unsqueeze(1)

        dot = torch.einsum("bqihpd,bkjhpd->bqjhp", q_exp, k_exp)
        cross = torch.norm(
            torch.cross(
                q_exp.unsqueeze(4),
                k_exp.unsqueeze(5),
                dim=-1
            ),
            dim=-1
        ).mean(dim=-1)

        norm_q = torch.norm(q_exp, dim=-1)
        norm_k = torch.norm(k_exp, dim=-1)
        curvature = cross / (norm_q * norm_k + self.eps)

        att = (dot - curvature).mean(dim=-1)

        return att.permute(0, 3, 1, 2)
    

class EnhancedIPA2(nn.Module):
    def __init__(self, c_s: int, c_z: int, c_hidden: int, 
                 no_heads: int, no_qk_points: int, no_v_points: int,
                 inf: float = 1e5, eps: float = 1e-8, dropout: float = 0.0):
        super().__init__()
        self.c_s, self.c_z, self.c_hidden = c_s, c_z, c_hidden
        self.no_heads, self.no_qk_points, self.no_v_points = no_heads, no_qk_points, no_v_points
        self.inf, self.eps, self.dropout = inf, eps, dropout

        # 标量注意力投影
        self.linear_q = nn.Linear(c_s, c_hidden * no_heads)
        self.linear_kv = nn.Linear(c_s, 2 * c_hidden * no_heads)

        # 几何点生成（合并坐标和方向投影）
        self.linear_q_points = self._init_point_proj(no_qk_points)
        self.linear_kv_points = self._init_point_proj(no_qk_points + no_v_points)
        
        # 动态几何门控（简化结构）
        self.point_gate = nn.Sequential(
            nn.Linear(c_s, no_heads * no_qk_points),
            nn.Sigmoid()
        )

        # 注意力参数合并
        self.geom_weight = nn.Parameter(torch.tensor([0.5, 0.5]))  # 位置/方向权重
        self.head_weights = nn.Parameter(torch.zeros(no_heads))

        # 输出层
        concat_dim = no_heads * (c_hidden + 7 * no_v_points + (c_z if c_z > 0 else 0))
        self.linear_out = nn.Linear(concat_dim, c_s)

        self._init_weights()

    def _init_point_proj(self, num_points: int) -> nn.Module:
        """统一初始化点投影层"""
        return nn.Sequential(
            nn.Linear(self.c_s, 6 * self.no_heads * num_points),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        nn.init.normal_(self.head_weights, std=0.02)
        nn.init.constant_(self.geom_weight, 0.5)

    def forward(self, s: torch.Tensor, r, z: Optional[torch.Tensor] = None, frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 标量投影 [优化1：合并KV投影]
        q = self.linear_q(s)
        k, v = self.linear_kv(s).chunk(2, dim=-1)
        
        # 几何点生成 [优化2：统一处理]
        q_pts = self._process_points(self.linear_q_points(s), self.no_qk_points, r)
        k_pts, v_pts = [self._process_points(proj, num, r) for proj, num in 
                       zip(self.linear_kv_points(s).chunk(2, dim=-1), 
                           [self.no_qk_points, self.no_v_points])]

        # 动态门控 [优化3：简化计算]
        q_pts = q_pts * rearrange(self.point_gate(s), '... (h p) -> ... h p 1', h=self.no_heads, p=self.no_qk_points)

        # 注意力计算 [优化4：合并标量和几何]
        att = self._compute_attention(q, k, q_pts, k_pts, frame_mask)
        
        # 特征聚合与输出
        return self.linear_out(self._aggregate_features(att, v, v_pts, z))

    def _process_points(self, pts: torch.Tensor, num_points: int, r) -> torch.Tensor:
        """统一处理几何点（合并坐标和方向变换） [优化5：减少reshape次数]"""
        pts = rearrange(pts, '... (h p d) -> ... h p d', h=self.no_heads, p=num_points, d=6)
        coords, dirs = pts[..., :3], pts[..., 3:]
        # 合并坐标和方向的变换计算
        transformed_coords = r[..., None, None].apply(coords)
        transformed_dirs = r[..., None, None].get_rots().apply(dirs)
        return torch.cat([transformed_coords, transformed_dirs], dim=-1)

    def _compute_attention(self, q, k, q_pts, k_pts, mask) -> torch.Tensor:
        """合并注意力计算流程 [优化6：简化计算步骤]"""
        # 标量注意力
        scalar_att = torch.einsum('bqhc,bkhc->bhqk', 
                                 rearrange(q, '... (h c) -> ... h c', h=self.no_heads), 
                                 rearrange(k, '... (h c) -> ... h c', h=self.no_heads)) / (self.c_hidden**0.5)
        
        q_coords = rearrange(q_pts[..., :3].mean(dim=-2), 
                    'b l h d -> b h l d')  # [B, H, L, 3]
        k_coords = rearrange(k_pts[..., :3].mean(dim=-2),
                    'b l h d -> b h l d')  # [B, H, L, 3]

        # 跨头距离矩阵
        pos_att = -torch.cdist(q_coords, k_coords, p=2)  # [B, H, L, L]
        
        # 几何注意力（简化版）
        dir_att = torch.einsum('bqhpk,blhpk->bhql', q_pts[..., 3:], k_pts[..., 3:])  # 方向点积
        
        # 合并注意力 [优化7：权重求和]
        att = scalar_att + self.geom_weight[0] * pos_att + self.geom_weight[1] * dir_att
        att = att * torch.sigmoid(self.head_weights.view(-1, 1, 1))  # 头权重
        
        # 掩码处理
        if mask is not None:
            att += self.inf * (1 - mask.unsqueeze(1) @ mask.unsqueeze(2))  # [B, 1, L, L]
            
        return F.softmax(att, dim=-1)

    def _aggregate_features(self, att, v, v_pts, z) -> torch.Tensor:
        """特征聚合优化"""
        # 标量特征
        o_scalar = torch.einsum('bhqk,bkhc->bqhc', att, 
                               rearrange(v, '... (h c) -> ... h c', h=self.no_heads))
        
        # 几何特征（简化逆变换）
        o_geom = self._fast_inverse_transform(
            torch.einsum('bhqk,bkhpd->bqhpd', att, v_pts), 
            v_pts  # 近似全局坐标系
        )
        
        # 拼接特征
        features = [rearrange(o_scalar, '... h c -> ... (h c)'), 
                   rearrange(o_geom, '... h p d -> ... (h p d)')]
        
        if self.c_z > 0:
            features.append(rearrange(torch.einsum('bhqk,bkzc->bqzhc', att, z), '... h c -> ... (h c)'))
            
        return torch.cat(features, dim=-1)

    def _fast_inverse_transform(self, geom: torch.Tensor, ref_points: torch.Tensor) -> torch.Tensor:
        """快速逆变换近似 [优化8：避免精确逆变换]"""
        # 使用参考点近似全局坐标系
        local_coords = geom[..., :3] - ref_points[..., :3].mean(dim=-2, keepdim=True)
        local_dirs = F.normalize(geom[..., 3:6], dim=-1)
        return torch.cat([local_coords, local_dirs, torch.norm(local_coords, dim=-1, keepdim=True)], dim=-1)
    
class EnhancedIPA3(nn.Module):
    def __init__(self, c_s: int, c_z: int, c_hidden: int, 
                 no_heads: int, no_qk_points: int, no_v_points: int,
                 inf: float = 1e5, eps: float = 1e-8, dropout: float = 0.0):
        super().__init__()
        self.c_s, self.c_z, self.c_hidden = c_s, c_z, c_hidden
        self.no_heads, self.no_qk_points, self.no_v_points = no_heads, no_qk_points, no_v_points
        self.inf, self.eps, self.dropout = inf, eps, dropout

        # 标量注意力投影
        self.linear_q = nn.Linear(c_s, c_hidden * no_heads)
        self.linear_kv = nn.Linear(c_s, 2 * c_hidden * no_heads)

        # 几何点生成（精确投影）
        self.linear_q_points = self._init_point_proj(no_qk_points)
        self.linear_kv_points = self._init_point_proj(no_qk_points + no_v_points)
        
        # 动态几何门控
        self.point_gate = nn.Sequential(
            nn.Linear(c_s, no_heads * no_qk_points),
            nn.Sigmoid()
        )

        # 几何注意力参数
        self.geom_weight = nn.Parameter(torch.tensor([0.5, 0.5]))  # 位置/方向权重
        self.head_weights = nn.Parameter(torch.zeros(no_heads))

        # 输出层
        concat_dim = no_heads * (c_hidden + 7 * no_v_points + (c_z if c_z > 0 else 0))
        self.linear_out = nn.Linear(concat_dim, c_s)

        self._init_weights()

    def _init_point_proj(self, num_points: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.c_s, 6 * self.no_heads * num_points),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        nn.init.normal_(self.head_weights, std=0.02)
        nn.init.constant_(self.geom_weight, 0.5)

    def forward(self, s: torch.Tensor, r, z: Optional[torch.Tensor] = None, 
               frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 标量投影
        q = self.linear_q(s)
        k, v = self.linear_kv(s).chunk(2, dim=-1)
        
        # 精确几何点处理
        q_pts = self._process_points(self.linear_q_points(s), self.no_qk_points, r)
        k_pts, v_pts = [self._process_points(proj, num, r) for proj, num in 
                        zip(self.linear_kv_points(s).chunk(2, dim=-1), 
                            [self.no_qk_points, self.no_v_points])]

        # 动态门控
        q_pts = q_pts * rearrange(self.point_gate(s), '... (h p) -> ... h p 1', 
                                 h=self.no_heads, p=self.no_qk_points)

        # 精确注意力计算
        att = self._compute_attention(q, k, q_pts, k_pts, frame_mask)
        
        # 特征聚合与输出
        return self.linear_out(self._aggregate_features(att, v, v_pts, z, r))

    def _process_points(self, pts: torch.Tensor, num_points: int, r) -> torch.Tensor:
        """精确几何点变换 (参考InvariantPointAttention)"""
        pts = rearrange(pts, '... (h p d) -> ... h p d', 
                       h=self.no_heads, p=num_points, d=6)
        coords, dirs = pts[..., :3], pts[..., 3:]
        
        # 分别应用刚体变换
        transformed_coords = r[..., None, None].apply(coords)  # 旋转+平移
        transformed_dirs = r[..., None, None].get_rots().apply(dirs)  # 仅旋转方向
        return torch.cat([transformed_coords, transformed_dirs], dim=-1)

    def _compute_attention(self, q, k, q_pts, k_pts, mask) -> torch.Tensor:
        """全点对几何注意力 (提升精度关键)"""
        # 标量注意力
        scalar_att = torch.einsum('bqhc,bkhc->bhqk', 
                                rearrange(q, '... (h c) -> ... h c', h=self.no_heads), 
                                rearrange(k, '... (h c) -> ... h c', h=self.no_heads)) / (self.c_hidden**0.5)
        
        # 全点对位置注意力
        delta = q_pts[..., :3].unsqueeze(2) - k_pts[..., :3].unsqueeze(1)  # [B, L_q, L_k, H, Pq, 3]
        pos_att = (-torch.sum(delta**2, dim=-1).mean(dim=(-1))).permute(0, 3, 1, 2) # [B, H, L_q, L_k]
        
        # 方向注意力 (点积 + 曲率)
        dot = torch.einsum('bqhpk,blhpk->bhql', q_pts[..., 3:], k_pts[..., 3:])
        cross = torch.norm(torch.cross(q_pts[..., 3:], k_pts[..., 3:], dim=-1), dim=-1)
        norm_q = torch.norm(q_pts[..., 3:], dim=-1)
        norm_k = torch.norm(k_pts[..., 3:], dim=-1)
        curvature = (cross / (norm_q * norm_k + self.eps)).mean(dim=-1)
        curvature = curvature.permute(0, 2, 1).unsqueeze(-1)  # [B, H, L, 1]
        dir_att = dot - curvature
        
        # 合并注意力
        att = scalar_att + self.geom_weight[0] * pos_att + self.geom_weight[1] * dir_att
        att = att * torch.sigmoid(self.head_weights.view(-1, 1, 1))
        
        # 精确掩码处理
        if mask is not None:
            mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # [B, L, L]
            att += self.inf * (1 - mask_2d.unsqueeze(1))       # [B, H, L, L]
            
        return F.softmax(att, dim=-1)

    def _aggregate_features(self, att, v, v_pts, z, r) -> torch.Tensor:
        """特征聚合 (含精确逆变换)"""
        # 标量特征
        o_scalar = torch.einsum('bhqk,bkhc->bqhc', att, 
                               rearrange(v, '... (h c) -> ... h c', h=self.no_heads))
        
        # 几何特征聚合与逆变换
        o_geom = torch.einsum('bhqk,bkhpd->bqhpd', att, v_pts)
        o_geom = self._exact_inverse_transform(o_geom, r)
        
        # 拼接特征
        features = [rearrange(o_scalar, '... h c -> ... (h c)'), 
                   rearrange(o_geom, '... h p d -> ... (h p d)')]
        
        if self.c_z > 0:
            o_pair = torch.einsum('bhqk,bkzc->bqzhc', att, z)
            features.append(rearrange(o_pair, '... h c -> ... (h c)'))
            
        return torch.cat(features, dim=-1)

    def _exact_inverse_transform(self, geom: torch.Tensor, r: Rigid) -> torch.Tensor:
        """精确刚体逆变换 (参考InvariantPointAttention)"""
        inv_rot = r.get_rots().invert()
        trans_inv = -inv_rot.apply(r.get_trans())
        
        # 坐标逆变换
        global_coords = geom[..., :3]
        local_coords = inv_rot[..., None, None].apply(global_coords) + trans_inv.unsqueeze(-2).unsqueeze(-2)
        
        # 方向逆变换
        global_dirs = geom[..., 3:6]
        local_dirs = inv_rot[..., None, None].apply(global_dirs)
        local_dirs = F.normalize(local_dirs, dim=-1, eps=self.eps)
        
        # 拼接特征
        return torch.cat([
            local_coords,
            local_dirs,
            torch.norm(local_coords, dim=-1, keepdim=True)
        ], dim=-1)

    def geometric_consistency_loss(self, pred_coords, true_coords):
        """几何一致性损失 (可选)"""
        return F.mse_loss(pred_coords, true_coords)