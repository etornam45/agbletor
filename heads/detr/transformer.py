# Use dinov3 vit-small as backbone
# for DETR (Decoder) detection model

from typing import Dict

import torch
import torch.nn as nn

from heads.detr.deformable_attn import DeformableDecoderLayer
from heads.detr.ffn import FFN


def build_2d_sincos_pos_embed(h: int, w: int, d_model: int) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij"
    )
    assert d_model % 4 == 0
    omega = torch.arange(d_model // 4, dtype=torch.float32) / (d_model // 4)
    omega = 1.0 / (10000**omega)
    y_enc = y.flatten()[:, None] * omega[None]
    x_enc = x.flatten()[:, None] * omega[None]
    pos = torch.cat(
        [torch.sin(x_enc), torch.cos(x_enc), torch.sin(y_enc), torch.cos(y_enc)],
        dim=1,
    )
    return pos[None]  # (1, N, d_model)


class DETRTDecoder(nn.Module):
    def __init__(
        self, d_model, n_heads, num_layers, dim_ff, n_points=4, dropout=0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DeformableDecoderLayer(d_model, n_heads, dim_ff, n_points, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, tgt, memory, h, w, tgt_mask=None):
        intermediates = []
        for layer in self.layers:
            tgt = layer(tgt, memory, h, w, tgt_mask)
            intermediates.append(tgt)
        return intermediates


class DETR(nn.Module):
    def __init__(
        self,
        d_model=384,
        n_heads=6,
        num_layers=6,
        dim_ff=1024,
        n_classes=20,
        n_queries=50,
        dropout=0.1,
        img_size=224,
        patch_size=16,
        n_points=4,
    ):
        super().__init__()
        self.query_embed = nn.Embedding(n_queries, d_model)
        self.decoder = DETRTDecoder(
            d_model, n_heads, num_layers, dim_ff, n_points, dropout
        )
        self.bbox_pred = FFN(d_model, d_model, 4, n_layers=2, dropout=dropout)
        self.class_pred = FFN(d_model, d_model, n_classes, n_layers=2, dropout=dropout)

        h = w = img_size // patch_size
        self.register_buffer("_pos", build_2d_sincos_pos_embed(h, w, d_model))

        self.n_queries = n_queries
        self.d_model = d_model

    def forward(self, img_embed, tgt_mask=None, memory_mask=None) -> Dict:
        B, N, _ = img_embed.shape
        h = w = int(N**0.5)
        img_embed = img_embed + self._pos

        query_ids = torch.arange(self.n_queries, device=img_embed.device)
        tgt = self.query_embed(query_ids).unsqueeze(0).expand(B, -1, -1)

        all_hs = self.decoder(tgt, img_embed, h, w, tgt_mask)

        final_hs = all_hs[-1]
        out = {
            "logits": self.class_pred(final_hs),
            "boxes": torch.sigmoid(self.bbox_pred(final_hs)),
        }

        out["aux"] = [
            {
                "logits": self.class_pred(hs),
                "boxes": torch.sigmoid(self.bbox_pred(hs)),
            }
            for hs in all_hs[:-1]
        ]

        return out


def build_detr(
    d_model: int = 384,
    n_heads: int = 6,
    num_layers: int = 6,
    dim_ff: int = 1024,
    n_classes: int = 20,
    n_queries: int = 100,
    n_points: int = 4,
    dropout: float = 0.1,
) -> DETR:
    return DETR(
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_ff=dim_ff,
        n_classes=n_classes,
        n_queries=n_queries,
        dropout=dropout,
        n_points=n_points,
    )


if __name__ == "__main__":
    import time

    from dinov3.utils.device import get_device

    device = get_device()
    model = build_detr(num_layers=4, n_classes=91, dropout=0.2, n_points=10).to(device)
    N = (224 // 16) ** 2
    print(f"n_patches: {N}")
    img_embed = torch.randn(1, N, 384, device=device)
    start = time.time()
    out = model(img_embed)
    print(time.time() - start)
    print("logits :", out["logits"].shape)
    print("boxes  :", out["boxes"].shape)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total / 1e6:.1f}M")
