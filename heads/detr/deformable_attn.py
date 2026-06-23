import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from heads.detr.ffn import FFN


def bilinear_sample(feat_grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Differentiable bilinear sampling (works on MPS where grid_sample backward is unsupported).

    Args:
        feat_grid:  (B, H, W, C)
        points:     (B, Q, K, 2)  — (x, y) normalised to [0, 1]
    Returns:
        sampled:    (B, Q, K, C)
    """
    B, H, W, C = feat_grid.shape
    _, Q, K, _ = points.shape

    x = points[..., 0] * (W - 1)
    y = points[..., 1] * (H - 1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = torch.clamp(x0 + 1, 0, W - 1)
    y1 = torch.clamp(y0 + 1, 0, H - 1)
    x0 = torch.clamp(x0, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)

    wx = (x - x0.float()).unsqueeze(-1)
    wy = (y - y0.float()).unsqueeze(-1)

    feat_flat = feat_grid.reshape(B, H * W, C)
    b_idx = torch.arange(B, device=feat_grid.device)[:, None, None].expand(B, Q, K)

    f00 = feat_flat[b_idx, y0 * W + x0]
    f01 = feat_flat[b_idx, y0 * W + x1]
    f10 = feat_flat[b_idx, y1 * W + x0]
    f11 = feat_flat[b_idx, y1 * W + x1]

    return (1 - wy) * ((1 - wx) * f00 + wx * f01) + wy * ((1 - wx) * f10 + wx * f11)


def grid_sample_points(feat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Differentiable bilinear sampling.

    Args:
        feat:   (B, C, H, W)
        points: (B, Q, K, 2) — (x, y) normalised to [0, 1]
    Returns:
        sampled: (B, Q, K, C)
    """
    B, C, H, W = feat.shape
    feat_nhwc = feat.permute(0, 2, 3, 1)
    if feat.device.type == "mps":
        return bilinear_sample(feat_nhwc, points)

    grid = points * 2.0 - 1.0
    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.permute(0, 2, 3, 1)


class DeformableCrossAttention(nn.Module):
    """
    Single-scale deformable cross-attention (Zhu et al., 2020).
    """

    def __init__(self, d_model: int, n_heads: int, n_points: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self._init_offsets()

    def _init_offsets(self):
        thetas = np.arange(self.n_heads) * (2 * np.pi / self.n_heads)
        grid = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
        grid /= np.abs(grid).max(-1, keepdims=True)
        grid = np.tile(grid[:, None, :], (1, self.n_points, 1))
        for j in range(self.n_points):
            grid[:, j, :] *= j + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(
                torch.tensor(grid.reshape(-1), dtype=torch.float32)
            )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        B, Q, _ = query.shape

        value = self.value_proj(memory)
        value = value.reshape(B, h, w, self.n_heads, self.head_dim)
        value = value.permute(0, 3, 4, 1, 2)  # (B, nH, hd, H, W)

        offsets = self.sampling_offsets(query)
        offsets = offsets.reshape(B, Q, self.n_heads, self.n_points, 2)
        offsets = offsets / torch.tensor([w, h], dtype=offsets.dtype, device=offsets.device)

        ref = reference_points[:, :, None, None, :]
        sample_pts = torch.clamp(ref + offsets, 0.0, 1.0)

        attn = self.attention_weights(query)
        attn = F.softmax(
            attn.reshape(B, Q, self.n_heads, self.n_points), dim=-1
        )

        head_out = []
        for hd in range(self.n_heads):
            v_h = value[:, hd]  # (B, hd, H, W)
            pts_h = sample_pts[:, :, hd, :, :]  # (B, Q, nP, 2)
            sampled = grid_sample_points(v_h, pts_h)  # (B, Q, nP, hd)
            w_h = attn[:, :, hd, :, None]
            head_out.append((sampled * w_h).sum(dim=2))

        out = torch.cat(head_out, dim=-1)
        return self.out_proj(out)


class DeformableDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        n_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = DeformableCrossAttention(d_model, n_heads, n_points)
        self.ffn = FFN(d_model, dim_ff, dropout=dropout, n_layers=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ref_point_head = nn.Linear(d_model, 2)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        h: int,
        w: int,
        tgt_mask=None,
    ) -> torch.Tensor:
        x = self.norm1(tgt)
        tgt2, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=False)
        tgt = tgt + self.dropout(tgt2)

        ref_pts = torch.sigmoid(self.ref_point_head(tgt))

        x = self.norm2(tgt)
        tgt = tgt + self.dropout(self.cross_attn(x, memory, ref_pts, h, w))

        tgt = tgt + self.dropout(self.ffn(self.norm3(tgt)))
        return tgt


if __name__ == "__main__":
    from dinov3.utils.device import get_device

    device = get_device()
    d_model = 384
    n_heads = 6
    n_points = 4
    dim_ff = 1024
    B = 2
    Q = 100
    H = W = 14

    decoder = nn.ModuleList(
        [
            DeformableDecoderLayer(d_model, n_heads, dim_ff, n_points).to(device)
            for _ in range(3)
        ]
    )

    query = torch.randn(B, Q, d_model, device=device)
    memory = torch.randn(B, H * W, d_model, device=device)
    for layer in decoder:
        query = layer(query, memory, H, W)
    print("Output shape:", query.shape)
