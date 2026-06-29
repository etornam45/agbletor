from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3.checkpoints.load import (
    ensure_backbone_checkpoint,
    load_checkpoint,
    validate_checkpoint_file,
)
from dinov3.models import vit_small
from heads.detr.transformer import build_2d_sincos_pos_embed
from heads.vqa.minicpm_loader import (
    MINICPM_MODEL_NAME,
    load_minicpm_llm,
    load_minicpm_tokenizer,
)

BACKBONE_WEIGHTS = "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DEFAULT_CHECKPOINT_DIR = "dinov3/checkpoints/model/vqa_minicpm"
VISION_DIM = 384
LLM_DIM = 1536
IMG_SIZE = 224
PATCH_SIZE = 16
GRID_SIZE = IMG_SIZE // PATCH_SIZE
NUM_VISUAL_TOKENS = 64
VISUAL_GRID = 8


class DINOv3MiniCPMHybrid(nn.Module):
    def __init__(
        self,
        vision_model: nn.Module,
        llm: nn.Module,
        num_visual_tokens: int = NUM_VISUAL_TOKENS,
        vision_dim: int = VISION_DIM,
        llm_dim: int = LLM_DIM,
        img_size: int = IMG_SIZE,
        patch_size: int = PATCH_SIZE,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.llm = llm
        self.num_visual_tokens = num_visual_tokens
        self.llm_dim = llm_dim

        self.vision_projection = nn.Linear(vision_dim, llm_dim)
        self.projection_norm = nn.LayerNorm(llm_dim)

        h = w = img_size // patch_size
        self.register_buffer(
            "_pos", build_2d_sincos_pos_embed(h, w, llm_dim)
        )

    def _llm_dtype(self) -> torch.dtype:
        return next(self.llm.parameters()).dtype

    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.vision_model(images, masks=None, is_training=True)
            patches = features["x_norm_patchtokens"]
        projected = self.projection_norm(self.vision_projection(patches))
        projected = projected + self._pos

        batch_size, _, dim = projected.shape
        grid = IMG_SIZE // PATCH_SIZE
        spatial = (
            projected.transpose(1, 2)
            .reshape(batch_size, dim, grid, grid)
        )
        pooled = F.interpolate(
            spatial,
            size=(VISUAL_GRID, VISUAL_GRID),
            mode="bilinear",
            align_corners=False,
        )
        return pooled.flatten(2).transpose(1, 2)

    def _merge_visual_and_text(
        self,
        visual_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ):
        text_embeds = self._get_text_embeddings(input_ids)
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        batch_size, visual_len, _ = visual_embeds.shape
        visual_mask = torch.ones(
            batch_size,
            visual_len,
            device=inputs_embeds.device,
            dtype=attention_mask.dtype if attention_mask is not None else torch.long,
        )
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape, device=input_ids.device, dtype=torch.long
            )
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        if labels is not None:
            visual_labels = torch.full(
                (batch_size, visual_len),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([visual_labels, labels], dim=1)

        return inputs_embeds, attention_mask, labels

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        visual_embeds = self.encode_visual(images)
        inputs_embeds, attention_mask, labels = self._merge_visual_and_text(
            visual_embeds, input_ids, attention_mask, labels
        )
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        num_beams: int = 1,
    ) -> torch.Tensor:
        visual_embeds = self.encode_visual(images)
        inputs_embeds, attention_mask, _ = self._merge_visual_and_text(
            visual_embeds, input_ids, attention_mask, labels=None
        )
        visual_len = visual_embeds.shape[1]
        visual_placeholders = torch.zeros(
            input_ids.shape[0],
            visual_len,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        full_input_ids = torch.cat([visual_placeholders, input_ids], dim=1)
        was_gc = getattr(self.llm, "is_gradient_checkpointing", False)
        if was_gc and hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()
        try:
            return self.llm.generate(
                input_ids=full_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                use_cache=True,
            )
        finally:
            if was_gc and hasattr(self.llm, "gradient_checkpointing_enable"):
                self.llm.gradient_checkpointing_enable()


def decode_generated_answer(
    tokenizer,
    gen_ids: torch.Tensor,
    prompt_len: int,
    num_visual_tokens: int = NUM_VISUAL_TOKENS,
) -> str:
    start = num_visual_tokens + prompt_len
    new_tokens = gen_ids[0, start:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_hybrid_model(
    device: torch.device,
    llm_model_name: str = MINICPM_MODEL_NAME,
    backbone_weights: str = BACKBONE_WEIGHTS,
    lora_r: int = 16,
    lora_alpha: int = 32,
    adapter_path: Optional[str] = None,
):
    vision_model = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    if Path(backbone_weights).exists() and not validate_checkpoint_file(
        backbone_weights, expected_sha256=None
    ):
        print(f"Warning: checkpoint at {backbone_weights} looks corrupt, re-downloading")
        Path(backbone_weights).unlink(missing_ok=True)

    backbone_weights = ensure_backbone_checkpoint(backbone_weights)
    load_checkpoint(vision_model, backbone_weights)
    vision_model.to(device)
    vision_model.eval()
    for param in vision_model.parameters():
        param.requires_grad = False

    llm = load_minicpm_llm(
        device,
        model_name=llm_model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        adapter_path=adapter_path,
    )

    model = DINOv3MiniCPMHybrid(
        vision_model=vision_model,
        llm=llm,
    ).to(device)

    if hasattr(model.llm, "enable_input_require_grads"):
        model.llm.enable_input_require_grads()

    tokenizer = load_minicpm_tokenizer(llm_model_name)
    return model, tokenizer


def adapter_trainable_params(model: DINOv3MiniCPMHybrid):
    return [p for p in model.llm.parameters() if p.requires_grad]


def vision_adapter_params(model: DINOv3MiniCPMHybrid):
    return list(model.vision_projection.parameters()) + list(
        model.projection_norm.parameters()
    )


def save_hybrid_checkpoint(model: DINOv3MiniCPMHybrid, checkpoint_dir: str) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(path / "adapter")
    torch.save(
        {
            "vision_projection": model.vision_projection.state_dict(),
            "projection_norm": model.projection_norm.state_dict(),
            "num_visual_tokens": model.num_visual_tokens,
        },
        path / "vision_adapter.pt",
    )


def load_hybrid_checkpoint(
    model: DINOv3MiniCPMHybrid,
    checkpoint_dir: str,
    device: torch.device,
    trainable_adapter: bool = False,
) -> None:
    from peft import PeftModel

    path = Path(checkpoint_dir)
    adapter_dir = path / "adapter"
    vision_path = path / "vision_adapter.pt"

    if adapter_dir.exists():
        if isinstance(model.llm, PeftModel):
            base_model = model.llm.get_base_model()
        else:
            base_model = model.llm
        model.llm = PeftModel.from_pretrained(
            base_model,
            str(adapter_dir),
            is_trainable=trainable_adapter,
        ).to(device)

    if vision_path.exists():
        state = torch.load(vision_path, map_location=device, weights_only=True)
        model.vision_projection.load_state_dict(state["vision_projection"])
        model.projection_norm.load_state_dict(state["projection_norm"])


if __name__ == "__main__":
    from dinov3.utils.device import get_device
    from heads.vqa.minicpm_loader import tokenize_chat_pair

    device = get_device()
    model, tokenizer = build_hybrid_model(device)

    images = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device)
    questions = [
        "What disease is affecting my maize plant?",
        "What can I do to treat it?",
    ]
    answers = [
        "Your maize has gray leaf spot with rectangular lesions.",
        "Apply fungicide and remove infected leaves.",
    ]

    input_ids = []
    labels = []
    for question, answer in zip(questions, answers):
        ids, lbls = tokenize_chat_pair(tokenizer, question, answer, max_length=128)
        input_ids.append(ids)
        labels.append(lbls)

    max_len = max(len(row) for row in input_ids)
    pad_id = tokenizer.pad_token_id
    input_ids_t = torch.full((2, max_len), pad_id, dtype=torch.long, device=device)
    labels_t = torch.full((2, max_len), -100, dtype=torch.long, device=device)
    attn = torch.zeros((2, max_len), dtype=torch.long, device=device)
    for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
        input_ids_t[i, : len(ids)] = torch.tensor(ids, device=device)
        labels_t[i, : len(lbls)] = torch.tensor(lbls, device=device)
        attn[i, : len(ids)] = 1

    out = model(images, input_ids_t, attn, labels=labels_t)
    print("loss:", out.loss.item())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable / 1e6:.1f}M")
