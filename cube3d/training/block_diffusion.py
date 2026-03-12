from __future__ import annotations

from contextlib import nullcontext
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder
from cube3d.model.gpt.block_diffusion_utils import (
    build_block_denoise_loss_mask,
    build_training_shape_attention_mask,
    duplicate_shape_position_ids,
    mask_shape_tokens,
    sample_block_timesteps,
    wrap_shape_attention_with_condition_prefix,
)
from cube3d.model.gpt.dual_stream_roformer import DualStreamRoformer

try:
    from safetensors.torch import load_model as load_safetensors_model
    from safetensors.torch import save_file as save_safetensors_file
except ImportError:
    load_safetensors_model = None
    save_safetensors_file = None


def load_config(cfg_path: str) -> DictConfig:
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg


def parse_structured(cfg_type: Any, cfg: DictConfig) -> Any:
    return OmegaConf.structured(cfg_type(**cfg))


def select_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def _extract_state_dict(checkpoint: Any, preferred_keys: tuple[str, ...]) -> Any:
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in preferred_keys:
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            return state_dict

    for key in ("model", "state_dict", "gpt_model", "shape_model"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            return state_dict

    return checkpoint


def load_model_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    preferred_keys: tuple[str, ...] = (),
) -> None:
    if ckpt_path.endswith(".safetensors"):
        if load_safetensors_model is None:
            raise ImportError(
                "safetensors is required to load .safetensors checkpoints."
            )
        load_safetensors_model(model, ckpt_path)
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint, preferred_keys)
    model.load_state_dict(state_dict)


@dataclass
class BlockDiffusionInputs:
    clean_shape_ids: torch.Tensor
    noisy_shape_ids: torch.Tensor
    shape_embed: torch.Tensor
    shape_position_ids: torch.Tensor
    attn_mask: torch.Tensor
    denoise_mask: torch.Tensor


class CubeBlockDiffusionTrainer:
    def __init__(
        self,
        config_path: str,
        shape_ckpt_path: str,
        device: Optional[torch.device] = None,
        gpt_ckpt_path: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        train_t_min: Optional[float] = None,
        train_t_max: Optional[float] = None,
        grad_clip_norm: float = 1.0,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.shape_ckpt_path = Path(shape_ckpt_path).expanduser().resolve()
        self.gpt_ckpt_path = (
            Path(gpt_ckpt_path).expanduser().resolve()
            if gpt_ckpt_path is not None
            else None
        )
        self.device = device or select_device()
        self.grad_clip_norm = grad_clip_norm
        self.autocast_enabled = self.device.type == "cuda"
        self._training_mask_cache: dict[tuple[int, int, str], torch.Tensor] = {}

        self.cfg = load_config(str(self.config_path))
        block_diffusion_cfg = self.cfg.get("block_diffusion")
        default_t_min = (
            float(block_diffusion_cfg.train_t_min)
            if block_diffusion_cfg is not None and "train_t_min" in block_diffusion_cfg
            else 0.45
        )
        default_t_max = (
            float(block_diffusion_cfg.train_t_max)
            if block_diffusion_cfg is not None and "train_t_max" in block_diffusion_cfg
            else 0.95
        )
        self.train_t_min = default_t_min if train_t_min is None else train_t_min
        self.train_t_max = default_t_max if train_t_max is None else train_t_max

        self.gpt_model = DualStreamRoformer(
            parse_structured(DualStreamRoformer.Config, self.cfg.gpt_model)
        ).to(self.device)
        if self.gpt_ckpt_path is not None:
            load_model_weights(
                self.gpt_model,
                str(self.gpt_ckpt_path),
                preferred_keys=("gpt_model", "model"),
            )

        self.shape_model = OneDAutoEncoder(
            parse_structured(OneDAutoEncoder.Config, self.cfg.shape_model)
        ).to(self.device)
        load_model_weights(
            self.shape_model,
            str(self.shape_ckpt_path),
            preferred_keys=("shape_model", "model"),
        )
        self.shape_model.eval()
        for parameter in self.shape_model.parameters():
            parameter.requires_grad_(False)

        self.text_model = CLIPTextModelWithProjection.from_pretrained(
            self.cfg.text_model_pretrained_model_name_or_path,
            force_download=False,
        ).eval()
        self.text_model.to(self.device)
        for parameter in self.text_model.parameters():
            parameter.requires_grad_(False)

        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(
            self.cfg.text_model_pretrained_model_name_or_path
        )

        self._copy_shape_codebook_to_gpt()

        self.gpt_model.train()
        self.mask_token_id = self.gpt_model.shape_mask_id
        self.block_size = self.gpt_model.cfg.block_size
        self.num_shape_tokens = self.shape_model.cfg.num_encoder_latents
        self.num_codes = self.shape_model.cfg.num_codes
        if getattr(self.gpt_model.cfg, "generation_mode", "ar") != "block_diffusion":
            logging.warning(
                "GPT config generation_mode is %s; training will still use "
                "forward_block_diffusion().",
                getattr(self.gpt_model.cfg, "generation_mode", "ar"),
            )

        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in self.gpt_model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

    def autocast_context(self):
        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def _copy_shape_codebook_to_gpt(self) -> None:
        with torch.no_grad():
            codebook = self.shape_model.bottleneck.block.get_codebook()
            codebook = self.gpt_model.shape_proj(codebook).detach()
        self.gpt_model.transformer.wte.weight.data[: codebook.shape[0]] = codebook

    def prepare_conditions_with_bbox(
        self,
        cond: torch.Tensor,
        bbox_xyz: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not hasattr(self.gpt_model, "bbox_proj"):
            return cond

        if bbox_xyz is None:
            bbox_xyz = torch.zeros((cond.shape[0], 3), dtype=cond.dtype, device=self.device)
        else:
            bbox_xyz = bbox_xyz.to(device=self.device, dtype=cond.dtype)

        bbox_embed = self.gpt_model.bbox_proj(bbox_xyz).unsqueeze(1)
        return torch.cat([cond, bbox_embed], dim=1)

    def encode_conditions(
        self,
        prompt_text: list[str],
        bbox_xyz: Optional[torch.Tensor],
    ) -> torch.Tensor:
        text_inputs = self.text_tokenizer(
            prompt_text,
            max_length=self.text_tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

        with torch.no_grad():
            encoded = self.text_model(**text_inputs)

        if self.gpt_model.cfg.use_pooled_text_embed:
            cond = encoded.text_embeds.unsqueeze(1)
        else:
            cond = encoded.last_hidden_state

        cond = self.gpt_model.encode_text(cond)
        return self.prepare_conditions_with_bbox(cond, bbox_xyz)

    @torch.no_grad()
    def encode_shapes(self, point_cloud: torch.Tensor) -> torch.Tensor:
        point_cloud = point_cloud.to(self.device, non_blocking=True)
        with self.autocast_context():
            encoded = self.shape_model.encode(point_cloud)
        return encoded[3]["indices"].long()

    def get_training_attention_mask(
        self,
        num_shape_tokens: int,
        cond_len: int,
    ) -> torch.Tensor:
        cache_key = (num_shape_tokens, cond_len, str(self.device))
        attn_mask = self._training_mask_cache.get(cache_key)
        if attn_mask is None:
            shape_mask = build_training_shape_attention_mask(
                num_shape_tokens=num_shape_tokens,
                block_size=self.block_size,
                device=self.device,
            )
            attn_mask = wrap_shape_attention_with_condition_prefix(shape_mask, cond_len)
            self._training_mask_cache[cache_key] = attn_mask
        return attn_mask

    def build_block_diffusion_inputs(
        self,
        cond: torch.Tensor,
        clean_shape_ids: torch.Tensor,
    ) -> BlockDiffusionInputs:
        batch_size, num_shape_tokens = clean_shape_ids.shape
        valid_mask = clean_shape_ids.ge(0) & clean_shape_ids.lt(self.num_codes)

        block_timesteps = sample_block_timesteps(
            batch_size=batch_size,
            num_shape_tokens=num_shape_tokens,
            block_size=self.block_size,
            t_min=self.train_t_min,
            t_max=self.train_t_max,
            device=clean_shape_ids.device,
        )
        noisy_shape_ids, _ = mask_shape_tokens(
            clean_shape_ids,
            block_timesteps=block_timesteps,
            mask_token_id=self.mask_token_id,
            valid_mask=valid_mask,
        )
        shape_input_ids = torch.cat([clean_shape_ids, noisy_shape_ids], dim=1)
        shape_embed = self.gpt_model.encode_token(shape_input_ids)
        shape_position_ids = duplicate_shape_position_ids(
            batch_size=batch_size,
            num_shape_tokens=num_shape_tokens,
            device=clean_shape_ids.device,
        )
        denoise_mask = build_block_denoise_loss_mask(
            noisy_shape_ids=noisy_shape_ids,
            mask_token_id=self.mask_token_id,
            valid_mask=valid_mask,
        )
        attn_mask = self.get_training_attention_mask(
            num_shape_tokens=num_shape_tokens,
            cond_len=cond.shape[1],
        )
        return BlockDiffusionInputs(
            clean_shape_ids=clean_shape_ids,
            noisy_shape_ids=noisy_shape_ids,
            shape_embed=shape_embed,
            shape_position_ids=shape_position_ids,
            attn_mask=attn_mask,
            denoise_mask=denoise_mask,
        )

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        bbox_xyz = batch["bbox_xyz"].to(self.device, non_blocking=True)

        cond = self.encode_conditions(batch["prompt_text"], bbox_xyz)
        clean_shape_ids = self.encode_shapes(batch["point_cloud"])

        diffusion_inputs = self.build_block_diffusion_inputs(cond, clean_shape_ids)

        with self.autocast_context():
            logits = self.gpt_model.forward_block_diffusion(
                embed=diffusion_inputs.shape_embed,
                cond=cond,
                attn_mask=diffusion_inputs.attn_mask,
                shape_position_ids=diffusion_inputs.shape_position_ids,
                use_single_blocks=False,
            )

        noisy_logits = logits[:, clean_shape_ids.shape[1] :, : self.num_codes]
        flat_mask = diffusion_inputs.denoise_mask.reshape(-1)
        flat_targets = clean_shape_ids.reshape(-1)
        flat_logits = noisy_logits.reshape(-1, self.num_codes)

        if bool(flat_mask.any().item()):
            masked_logits = flat_logits[flat_mask].float()
            masked_targets = flat_targets[flat_mask]
            loss = F.cross_entropy(masked_logits, masked_targets, reduction="mean")
        else:
            loss = flat_logits.sum() * 0.0

        denoise_tokens = int(flat_mask.sum().item())
        total_tokens = int(flat_mask.numel())
        metrics = {
            "loss": float(loss.detach().item()),
            "denoise_tokens": float(denoise_tokens),
            "mask_ratio": denoise_tokens / max(total_tokens, 1),
        }
        return loss, metrics

    def save_checkpoint(
        self,
        output_dir: str | Path,
        global_step: int,
        epoch: int,
        tag: Optional[str] = None,
    ) -> Path:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_tag = tag or f"step_{global_step:08d}"
        model_state = {
            key: value.detach().cpu()
            for key, value in self.gpt_model.state_dict().items()
        }
        if save_safetensors_file is not None:
            model_path = output_dir / f"gpt_{checkpoint_tag}.safetensors"
            save_safetensors_file(model_state, str(model_path))
        else:
            model_path = output_dir / f"gpt_{checkpoint_tag}.pt"
            torch.save(model_state, model_path)

        trainer_state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": self.optimizer.state_dict(),
            "config_path": str(self.config_path),
            "gpt_ckpt_path": str(self.gpt_ckpt_path) if self.gpt_ckpt_path else None,
            "shape_ckpt_path": str(self.shape_ckpt_path),
            "train_t_min": self.train_t_min,
            "train_t_max": self.train_t_max,
            "block_size": self.block_size,
            "mask_token_id": self.mask_token_id,
        }
        torch.save(
            trainer_state,
            output_dir / f"trainer_{checkpoint_tag}.pt",
        )
        return model_path

    def fit(
        self,
        dataloader: DataLoader,
        output_dir: str | Path,
        epochs: int = 1,
        max_steps: Optional[int] = None,
        save_interval: int = 500,
        log_interval: int = 10,
    ) -> Path:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(
            config=self.cfg,
            f=str(output_dir / "resolved_config.yaml"),
        )
        metadata = {
            "config_path": str(self.config_path),
            "gpt_ckpt_path": str(self.gpt_ckpt_path) if self.gpt_ckpt_path else None,
            "shape_ckpt_path": str(self.shape_ckpt_path),
            "train_t_min": self.train_t_min,
            "train_t_max": self.train_t_max,
            "block_size": self.block_size,
            "generation_mode": getattr(self.gpt_model.cfg, "generation_mode", "ar"),
        }
        (output_dir / "training_run.json").write_text(json.dumps(metadata, indent=2))

        global_step = 0
        last_epoch = 0
        last_checkpoint_path = output_dir / "gpt_final.pt"
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(1, epochs + 1):
            last_epoch = epoch
            progress = tqdm(dataloader, desc=f"epoch {epoch}", leave=False)
            for batch in progress:
                if max_steps is not None and global_step >= max_steps:
                    break

                self.gpt_model.train()
                loss, metrics = self.compute_loss(batch)
                loss.backward()

                grad_norm = None
                if self.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.gpt_model.parameters(), self.grad_clip_norm
                    )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

                progress.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    mask=f"{metrics['mask_ratio']:.3f}",
                )

                if global_step % max(log_interval, 1) == 0:
                    if grad_norm is None:
                        logging.info(
                            "step=%d epoch=%d loss=%.4f denoise_tokens=%d mask_ratio=%.3f",
                            global_step,
                            epoch,
                            metrics["loss"],
                            int(metrics["denoise_tokens"]),
                            metrics["mask_ratio"],
                        )
                    else:
                        logging.info(
                            "step=%d epoch=%d loss=%.4f denoise_tokens=%d mask_ratio=%.3f grad_norm=%.4f",
                            global_step,
                            epoch,
                            metrics["loss"],
                            int(metrics["denoise_tokens"]),
                            metrics["mask_ratio"],
                            float(grad_norm),
                        )

                if save_interval > 0 and global_step % save_interval == 0:
                    last_checkpoint_path = self.save_checkpoint(
                        output_dir=output_dir,
                        global_step=global_step,
                        epoch=epoch,
                    )

            if max_steps is not None and global_step >= max_steps:
                break

        last_checkpoint_path = self.save_checkpoint(
            output_dir=output_dir,
            global_step=global_step,
            epoch=last_epoch,
            tag="final",
        )
        return last_checkpoint_path
