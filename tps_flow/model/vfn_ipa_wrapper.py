"""
A wrapper class to make VFNScore compatible with IPA inputs.
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Sequence
from .vfn import ATTNTransition, GaussianLayer, Linear, VecLinear, flatten_final_dims, ipa_point_weights_init_, permute_final_dims

class VFN_InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
            self,
            ipa_conf,
            vfn_conf,
            inf: float = 1e5,
            eps: float = 1e-8,
    ):
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
        super(VFN_InvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf
        self._vfn_conf = vfn_conf

        self.c_s = ipa_conf.c_s
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_points = vfn_conf.no_points

        self.inf = inf
        self.eps = eps
        self.scale_pos = ipa_conf.coordinate_scaling
        self.gbf_k = vfn_conf.gbf_k
        self.g_dim = vfn_conf.g_dim

        self.vfn_attn_factor = vfn_conf.vfn_attn_factor
        self.dist_attn_factor = vfn_conf.dist_attn_factor
        self.attn_factor_sum = self.vfn_attn_factor + self.dist_attn_factor + 2.0

        hc = self.c_hidden * self.no_heads   # 32 * 4
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        self.no_v_points = ipa_conf.no_v_points
        concat_out_dim = self.c_hidden + self.no_v_points * 4
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.linear_points_q = Linear(self.c_s, self.no_points * 3)
        self.linear_points_v = Linear(self.c_s, self.no_points * 3)
        self.vlinear1 = VecLinear(self.no_points * 2, self.no_points)
        self.gbf = GaussianLayer(self.no_points, self.g_dim, K=self.gbf_k)
        self.attn_mlp = ATTNTransition(self.g_dim, self.no_heads)

        self.s_pt_layer_norm = nn.LayerNorm(self.c_s)

        self.scale_pos_fn = lambda x: x * ipa_conf.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos_fn)

        self.no_v_points = ipa_conf.no_v_points
        hpkv = self.no_heads * self.no_v_points * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

    def forward(
            self,
            s: torch.Tensor,
            r,
            mask: torch.Tensor,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """

        #######################################
        # interductive bias part
        #######################################
        coord = r.get_trans() * self.scale_pos
        pair_dist = coord[:, None, :, :] - coord[:, :, None, :]
        pair_dist = (pair_dist ** 2).sum(dim=-1)
        head_weights = self.softplus(self.head_weights)
        head_weights = head_weights * (math.sqrt(1.0 / (self.attn_factor_sum * (9.0 / 2)))) * (-0.5)
        dist_att = pair_dist[..., None] * head_weights[None, None, None, :]

        #######################################
        # IPA scalar attn part
        #######################################
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        #######################################
        # VFN part
        #######################################

        s_pt_normed = self.s_pt_layer_norm(s)
        q_pts = self.linear_points_q(s_pt_normed)
        v_pts = self.linear_points_v(s_pt_normed)
        v_pts = v_pts.reshape(v_pts.shape[:-1] + (self.no_points, 3))
        q_pts = q_pts.reshape(q_pts.shape[:-1] + (self.no_points, 3))

        r_paired = r[:, :, None].decompose(r[:, None, :])  # [dst, src, z]
        v_pts_local_paried = r_paired[:, :, :, None].apply(v_pts[:, None])  # TODO weian: rm torch.eye
        q_pts_repeated = q_pts[:, :, None].repeat(1, 1, q_pts.shape[1], 1, 1)
        vec_field = torch.cat([v_pts_local_paried, q_pts_repeated], dim=-2)
        vec_field = self.vlinear1(vec_field) + v_pts_local_paried
        vec_length = vec_field.norm(dim=-1)

        g = self.gbf(vec_length)

        vfn_attn = self.attn_mlp(g)

        ##########################
        # Compute attention scores
        ##########################
        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (self.attn_factor_sum * self.c_hidden))
        
        # if self.c_z > 0:
        #     # [*, N_res, N_res, H]
        #     b = self.linear_b(z[0])
        #     if (_offload_inference):
        #         z[0] = z[0].cpu()
        #     a += (math.sqrt(1.0 / self.attn_factor_sum) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # 1 is not padding; 0 is padding
        square_mask = self.inf * (square_mask - 1)

        a = a + (vfn_attn.permute(0, 3, 1, 2) / math.sqrt(self.attn_factor_sum)) * self.vfn_attn_factor
        a = a + dist_att.permute(0, 3, 1, 2) * self.dist_attn_factor
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]

        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        v_pts = self.linear_kv_points(s)
        v_pts = v_pts.reshape(v_pts.shape[:-1] + (self.no_v_points * self.no_heads, 3))
        r_scaled = self.scale_rigids(r)
        v_pts = r_scaled[..., None].apply(v_pts)
        v_pts = v_pts.reshape(v_pts.shape[:-2] + (self.no_heads, self.no_v_points, 3))

        o_pt = torch.sum(
            (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r_scaled[..., None, None].invert_apply(o_pt)
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # o_vfn = torch.einsum("bhds,bdsc->bdhc", a, g)
        # o_vfn = flatten_final_dims(o_vfn, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=o_feats.dtype)
        )

        return s, g

@dataclass
class IPAConfig:
    c_s: int = 384  # Single representation channel dimension
    c_hidden: int = 16  # Hidden channel dimension
    no_heads: int = 12  # Number of attention heads
    no_qk_points: int = 4  # Number of query/key points
    no_v_points: int = 8  # Number of value points
    coordinate_scaling: float = 50.0  # Coordinate scaling factor

@dataclass
class VFNConfig:
    no_points: int = 32  # Number of points for vector field
    gbf_k: int = 3  # Number of Gaussian basis functions
    g_dim: int = 32  # Dimension of Gaussian layer output
    vfn_attn_factor: float = 1.0  # VFN attention weight factor
    dist_attn_factor: float = 1.0  # Distance attention weight factor

class VFNScoreIPAWrapper(nn.Module):
    """
    Wrapper class that makes VFNScore compatible with IPA interface.
    """
    def __init__(
        self,
        c_s: int = 384,
        c_hidden: int = 16,
        no_heads: int = 12,
        no_qk_points: int = 4,
        no_v_points: int = 8,
        coordinate_scaling: float = 50.0,
        no_points: int = 32,
        gbf_k: int = 3,
        g_dim: int = 32,
        vfn_attn_factor: float = 1.0,
        dist_attn_factor: float = 1.0,
    ):
        """
        Args:
            c_s: Single representation channel dimension
            c_z: Pair representation channel dimension
            c_hidden: Hidden channel dimension
            no_heads: Number of attention heads
            no_qk_points: Number of query/key points
            no_v_points: Number of value points
            coordinate_scaling: Coordinate scaling factor
            no_points: Number of points for vector field
            gbf_k: Number of Gaussian basis functions
            g_dim: Dimension of Gaussian layer output
            vfn_attn_factor: VFN attention weight factor
            dist_attn_factor: Distance attention weight factor
        """
        super(VFNScoreIPAWrapper, self).__init__()

        # Create IPA config
        ipa_config = IPAConfig(
            c_s=c_s,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
            coordinate_scaling=coordinate_scaling,
        )

        # Create VFN config
        vfn_config = VFNConfig(
            no_points=no_points,
            gbf_k=gbf_k,
            g_dim=g_dim,
            vfn_attn_factor=vfn_attn_factor,
            dist_attn_factor=dist_attn_factor,
        )

        # Initialize the VFNScore
        self.vfn_score = VFN_InvariantPointAttention(
            ipa_conf=ipa_config,
            vfn_conf=vfn_config,
        )

    def forward(
        self,
        s: torch.Tensor,
        r,
        frame_mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass that maintains IPA interface.

        Args:
            s: [*, N_res, C_s] single representation
            z: [*, N_res, N_res, C_z] pair representation
            r: [*, N_res] transformation object
            mask: [*, N_res] mask
            _offload_inference: Whether to offload inference
            _z_reference_list: Optional list of reference tensors

        Returns:
            [*, N_res, C_s] Updated single representation
        """
        # Call VFNScore with IPA compatible inputs
        s_update, g = self.vfn_score(
            s=s,
            r=r,
            mask=frame_mask,
            _offload_inference=_offload_inference,
            _z_reference_list=_z_reference_list,
        )

        return s_update

def create_vfn_score_model(
    c_s: int = 384,
    c_z: int = 0,
    c_hidden: int = 32,
    no_heads: int = 4,
    no_qk_points: int = 8,
    no_v_points: int = 8,
    dropout: int = 0,
    coordinate_scaling: float = 50.0,
    no_points: int = 32,
    gbf_k: int = 3,
    g_dim: int = 32,
    vfn_attn_factor: float = 1.0,
    dist_attn_factor: float = 1.0,
) -> VFNScoreIPAWrapper:
    """
    Helper function to create a VFNScore model with IPA compatibility.

    Returns:
        Initialized VFNScoreIPAWrapper instance
    """
    return VFNScoreIPAWrapper(
        c_s=c_s,
        c_hidden=c_hidden,
        no_heads=no_heads,
        no_qk_points=no_qk_points,
        no_v_points=no_v_points,
        coordinate_scaling=coordinate_scaling,
        no_points=no_points,
        gbf_k=gbf_k,
        g_dim=g_dim,
        vfn_attn_factor=vfn_attn_factor,
        dist_attn_factor=dist_attn_factor,
    )