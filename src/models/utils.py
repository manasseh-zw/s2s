"""
Utilities for the owned Shona CSM training loop.

This module borrows the amortized decoder-loss setup from
`knottwill/sesame-finetune` while importing the actual CSM model math from the
official `SesameAILabs/csm` repository.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

load_dotenv()

MIMI_SAMPLE_RATE = int(os.getenv("MIMI_SAMPLE_RATE", "24000"))
TEXT_VOCAB_SIZE = int(os.getenv("TEXT_VOCAB_SIZE", "128256"))
AUDIO_VOCAB_SIZE = int(os.getenv("AUDIO_VOCAB_SIZE", "2051"))
AUDIO_NUM_CODEBOOKS = int(os.getenv("AUDIO_NUM_CODEBOOKS", "32"))

_CSM_COMPONENTS: dict[str, Any] | None = None


def get_csm_components() -> dict[str, Any]:
    global _CSM_COMPONENTS
    if _CSM_COMPONENTS is not None:
        return _CSM_COMPONENTS

    csm_repo_path = os.getenv("CSM_REPO_PATH")
    if not csm_repo_path:
        raise ValueError("CSM_REPO_PATH is not set")

    csm_repo = Path(csm_repo_path)
    if not (csm_repo / "models.py").exists():
        raise FileNotFoundError(f"Could not find CSM sources at {csm_repo}")

    repo_str = str(csm_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from generator import Generator, load_llama3_tokenizer, load_watermarker  # type: ignore
    from models import Model, ModelArgs, _create_causal_mask  # type: ignore

    _CSM_COMPONENTS = {
        "Generator": Generator,
        "Model": Model,
        "ModelArgs": ModelArgs,
        "_create_causal_mask": _create_causal_mask,
        "load_llama3_tokenizer": load_llama3_tokenizer,
        "load_watermarker": load_watermarker,
    }
    return _CSM_COMPONENTS


class WarmupDecayLR(LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        decay_type: str = "linear",
    ):
        self.warmup_steps = max(warmup_steps, 0)
        self.total_steps = max(total_steps, 1)
        self.decay_type = decay_type
        super().__init__(optimizer, self.lr_lambda, last_epoch=-1)

    def lr_lambda(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)

        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        decay_progress = max(step - self.warmup_steps, 0) / decay_steps

        if self.decay_type == "linear":
            return max(0.0, 1.0 - decay_progress)
        if self.decay_type == "constant":
            return 1.0
        if self.decay_type == "exponential":
            return 0.1 ** decay_progress
        if self.decay_type == "cosine":
            return float(0.5 * (1 + torch.cos(torch.pi * torch.tensor(decay_progress))))
        raise ValueError(f"Invalid decay type: {self.decay_type}")


def load_tokenizers(device: str | torch.device):
    components = get_csm_components()
    text_tokenizer = components["load_llama3_tokenizer"]()
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(AUDIO_NUM_CODEBOOKS)
    return text_tokenizer, mimi


def forward(self, tokens: torch.Tensor, tokens_mask: torch.Tensor) -> torch.Tensor:
    create_causal_mask = get_csm_components()["_create_causal_mask"]

    dtype = next(self.parameters()).dtype
    batch_size, seq_len, _ = tokens.size()
    device = tokens.device

    embeds = self._embed_tokens(tokens)

    audio_mask = tokens_mask[:, :, 0]
    target_tokens = tokens[audio_mask][:, :-1]
    c_embeds = embeds[:, :, :-1, :][audio_mask]

    masked_embeds = embeds * tokens_mask.unsqueeze(-1)
    hidden = masked_embeds.sum(dim=2)

    padding_mask = tokens_mask[:, :, 0] | tokens_mask[:, :, -1]
    backbone_attn_mask = create_causal_mask(seq_len, device)
    padding_3d = padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1)
    backbone_attn_mask = backbone_attn_mask.unsqueeze(0) * padding_3d
    backbone_attn_mask = backbone_attn_mask | torch.eye(
        seq_len,
        device=device,
        dtype=torch.bool,
    ).unsqueeze(0).expand(batch_size, -1, -1)

    input_pos = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    hidden = self.backbone(hidden, input_pos=input_pos, mask=backbone_attn_mask).to(dtype=dtype)

    audio_mask = torch.roll(audio_mask, -1, 1)
    audio_hidden = hidden[audio_mask]

    c0_logits = self.codebook0_head(audio_hidden)
    c0_target = target_tokens[:, 0]
    c0_loss = F.cross_entropy(c0_logits, c0_target)

    sample_count = max(c_embeds.size(0) // 16, 1)
    indices = torch.randperm(c_embeds.size(0), device=device)[:sample_count]
    c_embeds = c_embeds[indices][:, :-1, :]
    audio_hidden = audio_hidden[indices]
    target_tokens = target_tokens[indices][:, 1:]

    decoder_embeds = torch.cat([audio_hidden.unsqueeze(1), c_embeds], dim=1)
    sample_size, n_codebooks, _ = decoder_embeds.size()
    codebook_pos = torch.arange(0, n_codebooks, device=device).unsqueeze(0).expand(sample_size, n_codebooks)

    decoder_causal_mask = create_causal_mask(decoder_embeds.size(1), device).expand(sample_size, -1, -1)
    decoder_hidden = self.decoder(
        self.projection(decoder_embeds),
        input_pos=codebook_pos,
        mask=decoder_causal_mask,
    ).to(dtype=dtype)
    codebook_logits = torch.einsum("bsd,sdv->bsv", decoder_hidden[:, 1:, :], self.audio_head)

    decoder_loss = F.cross_entropy(
        codebook_logits.reshape(-1, codebook_logits.size(-1)),
        target_tokens.reshape(-1),
    )
    return 2 * (
        (1 - self.decoder_loss_weight) * c0_loss
        + self.decoder_loss_weight * decoder_loss
    )


def init_weights(model: nn.Module) -> nn.Module:
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    model.apply(_init_weights)
    nn.init.xavier_uniform_(model.audio_head)
    return model


def load_model(
    model_name_or_checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    decoder_loss_weight: float = 0.5,
):
    components = get_csm_components()
    model_cls = components["Model"]
    model_args_cls = components["ModelArgs"]

    if model_name_or_checkpoint_path is None or str(model_name_or_checkpoint_path).endswith(".pt"):
        config = model_args_cls(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=TEXT_VOCAB_SIZE,
            audio_vocab_size=AUDIO_VOCAB_SIZE,
            audio_num_codebooks=AUDIO_NUM_CODEBOOKS,
        )
        model = model_cls(config)
        if model_name_or_checkpoint_path:
            state_dict = torch.load(model_name_or_checkpoint_path, map_location="cpu")["model"]
            model.load_state_dict(state_dict)
        else:
            model = init_weights(model)
    else:
        model = model_cls.from_pretrained(str(model_name_or_checkpoint_path))

    model.decoder_loss_weight = decoder_loss_weight
    model.forward = types.MethodType(forward, model)
    return model.to(device=device)


def reset_caches(model) -> None:
    model.reset_caches()
    for module in model.modules():
        if hasattr(module, "cache_enabled"):
            module.cache_enabled = False
        if hasattr(module, "kv_cache"):
            module.kv_cache = None


def custom_generator_init(self, model, audio_tokenizer, text_tokenizer, watermarker) -> None:
    self._model = model
    self._model.setup_caches(1)
    self._text_tokenizer = text_tokenizer

    device = next(model.parameters()).device
    self._audio_tokenizer = audio_tokenizer.to(device=device)
    self.sample_rate = MIMI_SAMPLE_RATE
    self.device = device
    self._watermarker = watermarker


def generate_audio(
    model,
    audio_tokenizer,
    text_tokenizer,
    watermarker,
    text: str,
    speaker_id: int,
    device: str | torch.device,
    use_amp: bool = True,
    max_audio_length_ms: int = 10_000,
) -> np.ndarray:
    components = get_csm_components()
    generator_cls = components["Generator"]

    model.eval()
    generator_cls.__init__ = types.MethodType(custom_generator_init, generator_cls)
    generator = generator_cls(model, audio_tokenizer, text_tokenizer, watermarker)

    with torch.no_grad(), torch.amp.autocast(device_type=str(device), enabled=use_amp):
        audio = generator.generate(
            text=text,
            speaker=speaker_id,
            context=[],
            max_audio_length_ms=max_audio_length_ms,
        )
        audio = audio.squeeze().cpu().numpy()

    reset_caches(model)
    return audio


def load_watermarker(device: str | torch.device):
    return get_csm_components()["load_watermarker"](device=device)


def validate(model, val_loader, device: str | torch.device, use_amp: bool = True) -> float:
    model.eval()
    val_losses: list[float] = []

    with torch.no_grad(), torch.amp.autocast(device_type=str(device), enabled=use_amp):
        for tokens, tokens_mask in val_loader:
            tokens = tokens.to(device)
            tokens_mask = tokens_mask.to(device)
            val_losses.append(float(model(tokens, tokens_mask).item()))

    return sum(val_losses) / max(len(val_losses), 1)
