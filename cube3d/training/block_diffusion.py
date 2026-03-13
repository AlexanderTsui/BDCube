from __future__ import annotations

from contextlib import nullcontext
import json
import logging
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import trimesh
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

from cube3d.inference.logits_postprocesses import process_logits
from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder
from cube3d.model.gpt.block_diffusion_utils import (
    build_inference_shape_attention_mask,
    build_block_denoise_loss_mask,
    build_training_shape_attention_mask,
    duplicate_shape_position_ids,
    linear_noise_schedule,
    mask_keep_probability,
    mask_shape_tokens,
    sample_first_hitting_positions,
    sample_block_timesteps,
    wrap_shape_attention_with_condition_prefix,
)
from cube3d.model.gpt.dual_stream_roformer import DualStreamRoformer
from cube3d.training.data import SampleEvalSpec

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


def resolve_model_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported model dtype: {dtype_name}")


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

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, preferred_keys)
    model.load_state_dict(state_dict)


def peek_training_state(trainer_state_path: str | Path) -> dict[str, Any]:
    trainer_state = torch.load(
        Path(trainer_state_path), map_location="cpu", weights_only=False
    )
    if not isinstance(trainer_state, dict):
        raise ValueError(f"Trainer state at {trainer_state_path} is not a dictionary")
    return trainer_state


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    warmup_steps = max(0, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps == warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _to_cpu_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_state(item) for item in value)
    return value


def _normalize_path_str(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def _float_matches(lhs: Any, rhs: Any, atol: float = 1e-8) -> bool:
    return math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=atol)


def validate_training_state_compatibility(
    *,
    expected: dict[str, Any],
    loaded: dict[str, Any],
) -> None:
    mismatches: list[str] = []

    int_fields = (
        "block_size",
        "mask_token_id",
        "grad_accum_steps",
        "warmup_steps",
    )
    for field in int_fields:
        expected_value = expected.get(field)
        loaded_value = loaded.get(field)
        if expected_value is None or loaded_value is None:
            continue
        if int(expected_value) != int(loaded_value):
            mismatches.append(f"{field}: expected {expected_value}, got {loaded_value}")

    float_fields = ("train_t_min", "train_t_max", "min_lr_ratio")
    for field in float_fields:
        expected_value = expected.get(field)
        loaded_value = loaded.get(field)
        if expected_value is None or loaded_value is None:
            continue
        if not _float_matches(expected_value, loaded_value):
            mismatches.append(f"{field}: expected {expected_value}, got {loaded_value}")

    expected_amp_dtype = expected.get("amp_dtype")
    loaded_amp_dtype = loaded.get("amp_dtype")
    if expected_amp_dtype != loaded_amp_dtype:
        mismatches.append(
            f"amp_dtype: expected {expected_amp_dtype}, got {loaded_amp_dtype}"
        )

    expected_model_dtype = expected.get("model_dtype")
    loaded_model_dtype = loaded.get("model_dtype")
    if expected_model_dtype != loaded_model_dtype:
        mismatches.append(
            f"model_dtype: expected {expected_model_dtype}, got {loaded_model_dtype}"
        )

    expected_shape_ckpt = _normalize_path_str(expected.get("shape_ckpt_path"))
    loaded_shape_ckpt = _normalize_path_str(loaded.get("shape_ckpt_path"))
    if expected_shape_ckpt != loaded_shape_ckpt:
        mismatches.append(
            f"shape_ckpt_path: expected {expected_shape_ckpt}, got {loaded_shape_ckpt}"
        )

    expected_gpt_ckpt = _normalize_path_str(expected.get("gpt_ckpt_path"))
    if expected_gpt_ckpt is not None:
        loaded_gpt_candidates = {
            _normalize_path_str(loaded.get("gpt_ckpt_path")),
            _normalize_path_str(loaded.get("model_checkpoint_path")),
        }
        loaded_gpt_candidates.discard(None)
        if expected_gpt_ckpt not in loaded_gpt_candidates:
            mismatches.append(
                "gpt_ckpt_path: expected one of "
                f"{sorted(loaded_gpt_candidates)}, got {expected_gpt_ckpt}"
            )

    if mismatches:
        raise ValueError(
            "Training state is incompatible with the current trainer configuration: "
            + "; ".join(mismatches)
        )


def _is_out_of_memory_error(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(error).lower()


def _batch_debug_summary(batch: Optional[dict[str, Any]]) -> dict[str, Any]:
    if batch is None:
        return {}

    prompt_text = batch.get("prompt_text") or []
    mesh_paths = batch.get("mesh_path") or []
    return {
        "batch_size": len(prompt_text),
        "prompt_text": [str(text) for text in list(prompt_text)[:4]],
        "mesh_path": [str(path) for path in list(mesh_paths)[:4]],
    }


def _distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _materialize_file_alias(source_path: Path, target_path: Path) -> Path:
    source_path = Path(source_path).expanduser().resolve()
    target_path = Path(target_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)
    return target_path


class ExponentialMovingAverage:
    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow = {
            name: parameter.detach().cpu().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            shadow = self.shadow[name]
            shadow.mul_(self.decay).add_(parameter.detach().cpu(), alpha=1.0 - self.decay)

    def store(self, module: torch.nn.Module) -> None:
        self.backup = {
            name: parameter.detach().cpu().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    def copy_to(self, module: torch.nn.Module) -> None:
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter.data.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))

    def restore(self, module: torch.nn.Module) -> None:
        if not self.backup:
            return
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter.data.copy_(self.backup[name].to(device=parameter.device, dtype=parameter.dtype))
        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = float(state_dict["decay"])
        self.shadow = {
            name: tensor.detach().cpu().clone()
            for name, tensor in state_dict["shadow"].items()
        }
        self.backup = {}


@dataclass
class BlockDiffusionInputs:
    clean_shape_ids: torch.Tensor
    noisy_shape_ids: torch.Tensor
    shape_embed: torch.Tensor
    shape_position_ids: torch.Tensor
    attn_mask: torch.Tensor
    denoise_mask: torch.Tensor


@dataclass
class TrainerResumeState:
    global_step: int
    epoch: int
    resume_batch_idx: int
    best_val_loss: Optional[float]


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
        grad_accum_steps: int = 1,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.1,
        ema_decay: float = 0.0,
        amp_dtype: str = "bfloat16",
        model_dtype: str = "float32",
        use_grad_scaler: bool = True,
        activation_checkpointing: bool = False,
        offload_shape_model_to_cpu: bool = False,
        ddp_find_unused_parameters: bool = True,
        distributed_rank: int = 0,
        distributed_local_rank: int = 0,
        distributed_world_size: int = 1,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.shape_ckpt_path = Path(shape_ckpt_path).expanduser().resolve()
        self.gpt_ckpt_path = (
            Path(gpt_ckpt_path).expanduser().resolve()
            if gpt_ckpt_path is not None
            else None
        )
        self.device = device or select_device()
        self.distributed_rank = int(distributed_rank)
        self.distributed_local_rank = int(distributed_local_rank)
        self.distributed_world_size = int(distributed_world_size)
        self.distributed_enabled = self.distributed_world_size > 1
        self.ddp_find_unused_parameters = bool(ddp_find_unused_parameters)
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.shape_model_offload_to_cpu = bool(offload_shape_model_to_cpu)
        self._training_mask_cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self.model_dtype = resolve_model_dtype(model_dtype)
        self.model_dtype_name = model_dtype

        if amp_dtype not in {"bfloat16", "float16"}:
            raise ValueError(f"Unsupported amp_dtype {amp_dtype}")
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self.autocast_enabled = self.device.type == "cuda"
        scaler_enabled = bool(
            use_grad_scaler and self.device.type == "cuda" and self.amp_dtype == torch.float16
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.scheduler: Optional[LambdaLR] = None
        self.ema: Optional[ExponentialMovingAverage] = None

        self.cfg = load_config(str(self.config_path))
        self.cfg.gpt_model.activation_checkpointing = bool(activation_checkpointing)
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
        self.train_t_min = default_t_min if train_t_min is None else float(train_t_min)
        self.train_t_max = default_t_max if train_t_max is None else float(train_t_max)
        if not 0.0 <= self.train_t_min <= self.train_t_max <= 1.0:
            raise ValueError(
                f"Expected 0 <= train_t_min <= train_t_max <= 1, got "
                f"{self.train_t_min}, {self.train_t_max}"
            )
        self.default_num_diffusion_steps = (
            int(block_diffusion_cfg.num_diffusion_steps)
            if block_diffusion_cfg is not None and "num_diffusion_steps" in block_diffusion_cfg
            else 32
        )
        self.default_first_hitting = bool(
            block_diffusion_cfg.first_hitting
            if block_diffusion_cfg is not None and "first_hitting" in block_diffusion_cfg
            else False
        )
        self.default_first_hitting_tokens_per_step = (
            int(block_diffusion_cfg.first_hitting_tokens_per_step)
            if block_diffusion_cfg is not None and "first_hitting_tokens_per_step" in block_diffusion_cfg
            else 1
        )

        self.gpt_model = DualStreamRoformer(
            parse_structured(DualStreamRoformer.Config, self.cfg.gpt_model)
        ).to(device=self.device, dtype=self.model_dtype)
        if self.gpt_ckpt_path is not None:
            load_model_weights(
                self.gpt_model,
                str(self.gpt_ckpt_path),
                preferred_keys=("gpt_model", "model"),
            )

        shape_model_device = torch.device("cpu") if self.shape_model_offload_to_cpu else self.device
        self.shape_model = OneDAutoEncoder(
            parse_structured(OneDAutoEncoder.Config, self.cfg.shape_model)
        ).to(device=shape_model_device, dtype=self.model_dtype)
        load_model_weights(
            self.shape_model,
            str(self.shape_ckpt_path),
            preferred_keys=("shape_model", "model"),
        )
        self.shape_model_runtime_device = shape_model_device
        self.shape_model.eval()
        for parameter in self.shape_model.parameters():
            parameter.requires_grad_(False)

        self.text_model = CLIPTextModelWithProjection.from_pretrained(
            self.cfg.text_model_pretrained_model_name_or_path,
            force_download=False,
        ).eval()
        self.text_model.to(device=self.device, dtype=self.model_dtype)
        for parameter in self.text_model.parameters():
            parameter.requires_grad_(False)

        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(
            self.cfg.text_model_pretrained_model_name_or_path
        )

        self._copy_shape_codebook_to_gpt()
        self.gpt_train_model: torch.nn.Module
        if self.distributed_enabled:
            device_ids = [self.distributed_local_rank] if self.device.type == "cuda" else None
            output_device = self.distributed_local_rank if self.device.type == "cuda" else None
            self.gpt_train_model = DDP(
                self.gpt_model,
                device_ids=device_ids,
                output_device=output_device,
                broadcast_buffers=False,
                find_unused_parameters=self.ddp_find_unused_parameters,
            )
        else:
            self.gpt_train_model = self.gpt_model

        self.gpt_train_model.train()
        self.mask_token_id = self.gpt_model.shape_mask_id
        self.block_size = self.gpt_model.cfg.block_size
        self.num_shape_tokens = self.shape_model.cfg.num_encoder_latents
        self.num_codes = self.shape_model.cfg.num_codes
        if getattr(self.gpt_model.cfg, "generation_mode", "ar") != "block_diffusion":
            logging.warning(
                "GPT config generation_mode is %s; training will still use "
                "block-diffusion masks via the model forward pass.",
                getattr(self.gpt_model.cfg, "generation_mode", "ar"),
            )

        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in self.gpt_model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        if ema_decay > 0.0:
            self.ema = ExponentialMovingAverage(self.gpt_model, ema_decay)

    @property
    def is_main_process(self) -> bool:
        return self.distributed_rank == 0

    def distributed_barrier(self) -> None:
        if self.distributed_enabled and _distributed_is_initialized():
            if self.device.type == "cuda":
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()

    def _distributed_sum(self, *values: float) -> list[float]:
        packed = torch.tensor(values, device=self.device, dtype=torch.float64)
        if self.distributed_enabled and _distributed_is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        return [float(item) for item in packed.tolist()]

    def _distributed_max(self, *values: float) -> list[float]:
        packed = torch.tensor(values, device=self.device, dtype=torch.float64)
        if self.distributed_enabled and _distributed_is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.MAX)
        return [float(item) for item in packed.tolist()]

    def _reduce_step_metrics(
        self,
        *,
        loss_weighted_sum: float,
        denoise_tokens: float,
        total_tokens: float,
        batch_size: float,
    ) -> dict[str, float]:
        (
            loss_weighted_sum,
            denoise_tokens,
            total_tokens,
            batch_size,
        ) = self._distributed_sum(
            loss_weighted_sum,
            denoise_tokens,
            total_tokens,
            batch_size,
        )
        loss = 0.0
        if denoise_tokens > 0:
            loss = loss_weighted_sum / denoise_tokens
        mask_ratio = 0.0
        if total_tokens > 0:
            mask_ratio = denoise_tokens / total_tokens
        return {
            "loss": loss,
            "mask_ratio": mask_ratio,
            "denoise_tokens": denoise_tokens,
            "total_tokens": total_tokens,
            "batch_size": batch_size,
        }

    def autocast_context(self):
        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _training_state_signature(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "mask_token_id": self.mask_token_id,
            "train_t_min": self.train_t_min,
            "train_t_max": self.train_t_max,
            "grad_accum_steps": self.grad_accum_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "amp_dtype": "bfloat16" if self.amp_dtype == torch.bfloat16 else "float16",
            "model_dtype": self.model_dtype_name,
            "shape_ckpt_path": str(self.shape_ckpt_path),
            "gpt_ckpt_path": str(self.gpt_ckpt_path) if self.gpt_ckpt_path else None,
        }

    def _copy_shape_codebook_to_gpt(self) -> None:
        with torch.no_grad():
            codebook = self.shape_model.bottleneck.block.get_codebook().to(
                device=self.device,
                dtype=self.gpt_model.shape_proj.weight.dtype,
            )
            codebook = self.gpt_model.shape_proj(codebook).detach()
        self.gpt_model.transformer.wte.weight.data[: codebook.shape[0]] = codebook

    def _move_shape_model(self, device: torch.device | str) -> None:
        target_device = torch.device(device)
        if self.shape_model_runtime_device == target_device:
            return

        previous_device = self.shape_model_runtime_device
        self.shape_model.to(target_device)
        self.shape_model_runtime_device = target_device
        self.shape_model.eval()
        if previous_device.type == "cuda" and target_device.type == "cpu":
            torch.cuda.empty_cache()

    def _ensure_shape_model_for_runtime(self) -> None:
        self._move_shape_model(self.device)

    def _restore_shape_model_home(self) -> None:
        if self.shape_model_offload_to_cpu:
            self._move_shape_model(torch.device("cpu"))

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
        prompt_text: Optional[list[str]],
        bbox_xyz: Optional[torch.Tensor],
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_input_ids is not None and text_attention_mask is not None:
            text_inputs = {
                "input_ids": text_input_ids.to(self.device, non_blocking=True),
                "attention_mask": text_attention_mask.to(self.device, non_blocking=True),
            }
        else:
            if prompt_text is None:
                raise ValueError(
                    "encode_conditions requires prompt_text or pretokenized text_input_ids/attention_mask."
                )
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
        self._ensure_shape_model_for_runtime()
        try:
            with self.autocast_context():
                encoded = self.shape_model.encode(point_cloud)
            shape_ids = encoded[3]["indices"].long()
            if shape_ids.shape[1] != self.num_shape_tokens:
                raise ValueError(
                    f"Expected {self.num_shape_tokens} shape tokens, got {shape_ids.shape[1]}"
                )
            return shape_ids
        finally:
            self._restore_shape_model_home()

    @torch.no_grad()
    def encode_shape_tokens(self, shape_ids: torch.Tensor) -> torch.Tensor:
        return self.gpt_model.encode_token(shape_ids)

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

        if shape_position_ids.shape[1] != shape_input_ids.shape[1]:
            raise ValueError(
                "shape_position_ids and shape_input_ids must have the same sequence length"
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

        text_input_ids = batch.get("text_input_ids")
        text_attention_mask = batch.get("text_attention_mask")
        cond = self.encode_conditions(
            batch.get("prompt_text"),
            bbox_xyz,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
        )
        if batch.get("shape_ids") is not None:
            clean_shape_ids = batch["shape_ids"].to(self.device, non_blocking=True).long()
            if clean_shape_ids.shape[1] != self.num_shape_tokens:
                raise ValueError(
                    f"Expected {self.num_shape_tokens} precomputed shape tokens, got {clean_shape_ids.shape[1]}"
                )
        else:
            clean_shape_ids = self.encode_shapes(batch["point_cloud"])

        diffusion_inputs = self.build_block_diffusion_inputs(cond, clean_shape_ids)

        with self.autocast_context():
            logits = self.gpt_train_model(
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
            "total_tokens": float(total_tokens),
            "mask_ratio": denoise_tokens / max(total_tokens, 1),
            "batch_size": float(clean_shape_ids.shape[0]),
        }
        return loss, metrics

    def _state_dict_for_saving(self, use_ema: bool = False) -> dict[str, torch.Tensor]:
        if use_ema and self.ema is not None:
            self.ema.store(self.gpt_model)
            self.ema.copy_to(self.gpt_model)
        try:
            return {
                key: value.detach().cpu()
                for key, value in self.gpt_model.state_dict().items()
            }
        finally:
            if use_ema and self.ema is not None:
                self.ema.restore(self.gpt_model)

    def model_checkpoint_path(self, checkpoint_dir: str | Path, tag: str) -> Path:
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        suffix = ".safetensors" if save_safetensors_file is not None else ".pt"
        return checkpoint_dir / f"gpt_{tag}{suffix}"

    def save_model_checkpoint(
        self,
        checkpoint_dir: str | Path,
        tag: str,
        use_ema: bool = False,
    ) -> Path:
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_state = self._state_dict_for_saving(use_ema=use_ema)
        model_path = self.model_checkpoint_path(checkpoint_dir, tag)
        if save_safetensors_file is not None:
            save_safetensors_file(model_state, str(model_path))
        else:
            torch.save(model_state, model_path)
        return model_path

    def save_training_state(
        self,
        checkpoint_dir: str | Path,
        tag: str,
        global_step: int,
        epoch: int,
        resume_batch_idx: int,
        best_val_loss: Optional[float],
        model_checkpoint_path: str | Path,
    ) -> Path:
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainer_state = {
            "epoch": epoch,
            "global_step": global_step,
            "resume_batch_idx": resume_batch_idx,
            "best_val_loss": best_val_loss,
            "optimizer": _to_cpu_state(self.optimizer.state_dict()),
            "scheduler": _to_cpu_state(self.scheduler.state_dict())
            if self.scheduler is not None
            else None,
            "scaler": _to_cpu_state(self.scaler.state_dict()) if self.scaler.is_enabled() else None,
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "config_path": str(self.config_path),
            "gpt_ckpt_path": str(self.gpt_ckpt_path) if self.gpt_ckpt_path else None,
            "shape_ckpt_path": str(self.shape_ckpt_path),
            "train_t_min": self.train_t_min,
            "train_t_max": self.train_t_max,
            "block_size": self.block_size,
            "mask_token_id": self.mask_token_id,
            "grad_accum_steps": self.grad_accum_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "amp_dtype": "bfloat16" if self.amp_dtype == torch.bfloat16 else "float16",
            "model_dtype": self.model_dtype_name,
            "model_checkpoint_path": str(model_checkpoint_path),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        trainer_state_path = checkpoint_dir / f"trainer_{tag}.pt"
        torch.save(trainer_state, trainer_state_path)
        return trainer_state_path

    def load_training_state(self, trainer_state_path: str | Path) -> TrainerResumeState:
        trainer_state = peek_training_state(trainer_state_path)
        validate_training_state_compatibility(
            expected=self._training_state_signature(),
            loaded=trainer_state,
        )

        optimizer_state = trainer_state.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = trainer_state.get("scheduler")
        if scheduler_state is not None:
            if self.scheduler is None:
                raise RuntimeError("Scheduler must be created before loading training state")
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = trainer_state.get("scaler")
        if scaler_state is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(scaler_state)

        ema_state = trainer_state.get("ema")
        if ema_state is not None and self.ema is not None:
            self.ema.load_state_dict(ema_state)

        python_random_state = trainer_state.get("python_random_state")
        if python_random_state is not None:
            random.setstate(python_random_state)

        numpy_random_state = trainer_state.get("numpy_random_state")
        if numpy_random_state is not None:
            np.random.set_state(numpy_random_state)

        torch_random_state = trainer_state.get("torch_random_state")
        if torch_random_state is not None:
            torch.random.set_rng_state(torch_random_state)

        cuda_random_state = trainer_state.get("cuda_random_state")
        if cuda_random_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_random_state)

        return TrainerResumeState(
            global_step=int(trainer_state.get("global_step", 0)),
            epoch=int(trainer_state.get("epoch", 1)),
            resume_batch_idx=int(trainer_state.get("resume_batch_idx", 0)),
            best_val_loss=trainer_state.get("best_val_loss"),
        )

    def _try_save_latest_model_checkpoint(
        self,
        checkpoint_dir: Path,
    ) -> tuple[Optional[str], Optional[str]]:
        if not self.is_main_process:
            return None, "skipped_on_non_main_process"
        try:
            model_path = self.save_model_checkpoint(
                checkpoint_dir=checkpoint_dir,
                tag="latest",
            )
            return str(model_path), None
        except Exception as save_error:
            logging.exception("Failed to save emergency latest model checkpoint")
            return None, repr(save_error)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> dict[str, float]:
        if self.ema is not None:
            self.ema.store(self.gpt_model)
            self.ema.copy_to(self.gpt_model)

        was_training = self.gpt_train_model.training
        self.gpt_train_model.eval()

        weighted_loss_sum = 0.0
        total_denoise_tokens = 0.0
        total_tokens = 0.0
        num_batches = 0
        skipped_batches = 0

        try:
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                if batch is None:
                    skipped_batches += 1
                    continue

                loss, metrics = self.compute_loss(batch)
                denoise_tokens = float(metrics["denoise_tokens"])
                weighted_loss_sum += float(loss.detach().item()) * denoise_tokens
                total_denoise_tokens += denoise_tokens
                total_tokens += float(metrics["total_tokens"])
                num_batches += 1
        finally:
            if self.ema is not None:
                self.ema.restore(self.gpt_model)
            if was_training:
                self.gpt_train_model.train()

        (
            weighted_loss_sum,
            total_denoise_tokens,
            total_tokens,
            num_batches_total,
            skipped_batches_total,
        ) = self._distributed_sum(
            weighted_loss_sum,
            total_denoise_tokens,
            total_tokens,
            float(num_batches),
            float(skipped_batches),
        )

        num_batches = int(num_batches_total)
        skipped_batches = int(skipped_batches_total)
        if num_batches == 0:
            return {
                "loss": float("nan"),
                "mask_ratio": float("nan"),
                "denoise_tokens": 0.0,
                "num_batches": 0.0,
                "skipped_batches": float(skipped_batches),
            }

        return {
            "loss": weighted_loss_sum / max(total_denoise_tokens, 1.0),
            "mask_ratio": total_denoise_tokens / max(total_tokens, 1.0),
            "denoise_tokens": total_denoise_tokens,
            "num_batches": float(num_batches),
            "skipped_batches": float(skipped_batches),
        }

    def _reset_cuda_peak_memory_stats(self) -> None:
        if self.device.type != "cuda":
            return
        torch.cuda.reset_peak_memory_stats(self.device)

    def _cuda_peak_memory_stats_mb(self) -> dict[str, float]:
        if self.device.type != "cuda":
            return {
                "max_memory_allocated_mb": 0.0,
                "max_memory_reserved_mb": 0.0,
            }

        torch.cuda.synchronize(self.device)
        allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
        reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024.0**2)
        allocated_mb, reserved_mb = self._distributed_max(allocated_mb, reserved_mb)
        return {
            "max_memory_allocated_mb": allocated_mb,
            "max_memory_reserved_mb": reserved_mb,
        }

    @torch.no_grad()
    def sample_shape_ids(
        self,
        prompt_text: list[str],
        bbox_xyz: Optional[torch.Tensor] = None,
        num_diffusion_steps: Optional[int] = None,
        top_p: Optional[float] = None,
        first_hitting: Optional[bool] = None,
        first_hitting_tokens_per_step: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = len(prompt_text)
        if batch_size <= 0:
            raise ValueError("prompt_text must contain at least one prompt")

        num_steps = (
            self.default_num_diffusion_steps
            if num_diffusion_steps is None
            else int(num_diffusion_steps)
        )
        use_first_hitting = (
            self.default_first_hitting if first_hitting is None else bool(first_hitting)
        )
        tokens_per_step = (
            self.default_first_hitting_tokens_per_step
            if first_hitting_tokens_per_step is None
            else int(first_hitting_tokens_per_step)
        )
        if use_first_hitting and tokens_per_step <= 0:
            raise ValueError(
                "first_hitting_tokens_per_step must be positive when first_hitting is enabled"
            )

        cond = self.encode_conditions(prompt_text, bbox_xyz)
        shape_ids = torch.full(
            (batch_size, self.num_shape_tokens),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )

        for block_start in range(0, self.num_shape_tokens, self.block_size):
            block_end = min(block_start + self.block_size, self.num_shape_tokens)
            curr_block_size = block_end - block_start

            if use_first_hitting:
                while shape_ids[:, block_start:block_end].eq(self.mask_token_id).any():
                    block_ids = shape_ids[:, block_start:block_end]
                    shape_input_ids = torch.cat([shape_ids[:, :block_start], block_ids], dim=1)
                    shape_embed = self.encode_shape_tokens(shape_input_ids)
                    shape_position_ids = torch.arange(
                        shape_input_ids.shape[1], device=self.device, dtype=torch.long
                    ).unsqueeze(0).expand(batch_size, -1)
                    shape_mask = build_inference_shape_attention_mask(
                        context_len=block_start,
                        block_len=curr_block_size,
                        device=self.device,
                    )
                    attn_mask = wrap_shape_attention_with_condition_prefix(
                        shape_mask, cond.shape[1]
                    )
                    logits = self.gpt_model.forward_block_diffusion(
                        embed=shape_embed,
                        cond=cond,
                        attn_mask=attn_mask,
                        shape_position_ids=shape_position_ids,
                    )
                    block_logits = logits[:, -curr_block_size:, : self.num_codes]
                    candidate_ids = process_logits(block_logits, top_p=top_p).squeeze(-1)
                    mask_positions = block_ids.eq(self.mask_token_id)
                    update_mask = sample_first_hitting_positions(
                        mask_positions, tokens_per_step
                    )
                    block_ids = torch.where(update_mask, candidate_ids, block_ids)
                    shape_ids[:, block_start:block_end] = block_ids
                continue

            step_timesteps = [
                linear_noise_schedule(step_idx, num_steps)
                for step_idx in range(num_steps + 1)
            ]
            for step_idx in range(num_steps):
                block_ids = shape_ids[:, block_start:block_end]
                if not block_ids.eq(self.mask_token_id).any():
                    break

                shape_input_ids = torch.cat([shape_ids[:, :block_start], block_ids], dim=1)
                shape_embed = self.encode_shape_tokens(shape_input_ids)
                shape_position_ids = torch.arange(
                    shape_input_ids.shape[1], device=self.device, dtype=torch.long
                ).unsqueeze(0).expand(batch_size, -1)
                shape_mask = build_inference_shape_attention_mask(
                    context_len=block_start,
                    block_len=curr_block_size,
                    device=self.device,
                )
                attn_mask = wrap_shape_attention_with_condition_prefix(
                    shape_mask, cond.shape[1]
                )
                logits = self.gpt_model.forward_block_diffusion(
                    embed=shape_embed,
                    cond=cond,
                    attn_mask=attn_mask,
                    shape_position_ids=shape_position_ids,
                )
                block_logits = logits[:, -curr_block_size:, : self.num_codes]
                candidate_ids = process_logits(block_logits, top_p=top_p).squeeze(-1)

                current_t = step_timesteps[step_idx]
                next_t = step_timesteps[step_idx + 1]
                keep_probability = mask_keep_probability(current_t, next_t)
                mask_positions = block_ids.eq(self.mask_token_id)
                keep_mask = torch.rand_like(block_ids, dtype=torch.float32) < keep_probability
                keep_mask = keep_mask & mask_positions
                update_mask = (~keep_mask) & mask_positions
                block_ids = torch.where(update_mask, candidate_ids, block_ids)
                shape_ids[:, block_start:block_end] = block_ids

            remaining_mask = shape_ids[:, block_start:block_end].eq(self.mask_token_id)
            if remaining_mask.any():
                block_ids = shape_ids[:, block_start:block_end]
                shape_input_ids = torch.cat([shape_ids[:, :block_start], block_ids], dim=1)
                shape_embed = self.encode_shape_tokens(shape_input_ids)
                shape_position_ids = torch.arange(
                    shape_input_ids.shape[1], device=self.device, dtype=torch.long
                ).unsqueeze(0).expand(batch_size, -1)
                shape_mask = build_inference_shape_attention_mask(
                    context_len=block_start,
                    block_len=curr_block_size,
                    device=self.device,
                )
                attn_mask = wrap_shape_attention_with_condition_prefix(
                    shape_mask, cond.shape[1]
                )
                logits = self.gpt_model.forward_block_diffusion(
                    embed=shape_embed,
                    cond=cond,
                    attn_mask=attn_mask,
                    shape_position_ids=shape_position_ids,
                )
                block_logits = logits[:, -curr_block_size:, : self.num_codes]
                candidate_ids = process_logits(block_logits, top_p=top_p).squeeze(-1)
                block_ids = torch.where(remaining_mask, candidate_ids, block_ids)
                shape_ids[:, block_start:block_end] = block_ids

        return shape_ids

    @torch.no_grad()
    def decode_shape_ids(
        self,
        shape_ids: torch.Tensor,
        resolution_base: float = 8.0,
        chunk_size: int = 100_000,
    ) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
        self._ensure_shape_model_for_runtime()
        try:
            shape_ids = shape_ids.to(self.shape_model_runtime_device, non_blocking=True)
            clamped_shape_ids = (
                shape_ids[:, : self.shape_model.cfg.num_encoder_latents]
                .clamp(0, self.shape_model.cfg.num_codes - 1)
                .view(-1, self.shape_model.cfg.num_encoder_latents)
            )
            latents = self.shape_model.decode_indices(clamped_shape_ids)
            mesh_v_f, _ = self.shape_model.extract_geometry(
                latents,
                resolution_base=resolution_base,
                chunk_size=chunk_size,
                use_warp=True,
            )
            return mesh_v_f
        finally:
            self._restore_shape_model_home()

    def _safe_filename_slug(self, value: str, max_length: int = 80) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        if not slug:
            return "sample"
        return slug[:max_length].rstrip("-") or "sample"

    @torch.no_grad()
    def run_sample_generation_eval(
        self,
        sample_eval_specs: list[SampleEvalSpec],
        output_dir: str | Path,
        global_step: int,
        epoch: int,
        resolution_base: float = 8.0,
        chunk_size: int = 100_000,
        top_p: Optional[float] = None,
        num_diffusion_steps: Optional[int] = None,
        first_hitting: Optional[bool] = None,
        first_hitting_tokens_per_step: Optional[int] = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        samples_dir = (
            Path(output_dir).expanduser().resolve() / "samples" / f"step_{global_step:08d}"
        )
        samples_dir.mkdir(parents=True, exist_ok=True)
        sample_log_path = samples_dir / "samples.jsonl"

        requested_samples = len(sample_eval_specs)
        num_steps = (
            self.default_num_diffusion_steps
            if num_diffusion_steps is None
            else int(num_diffusion_steps)
        )
        use_first_hitting = (
            self.default_first_hitting if first_hitting is None else bool(first_hitting)
        )
        tokens_per_step = (
            self.default_first_hitting_tokens_per_step
            if first_hitting_tokens_per_step is None
            else int(first_hitting_tokens_per_step)
        )

        if use_first_hitting and tokens_per_step <= 0:
            raise ValueError(
                "sample eval requires first_hitting_tokens_per_step > 0 when first_hitting is enabled"
            )

        if self.ema is not None:
            self.ema.store(self.gpt_model)
            self.ema.copy_to(self.gpt_model)

        was_training = self.gpt_train_model.training
        self.gpt_train_model.eval()

        generated_samples = 0
        failed_samples = 0
        sample_records: list[dict[str, Any]] = []

        try:
            for sample_idx, spec in enumerate(sample_eval_specs):
                sample_name = (
                    f"{sample_idx:03d}_"
                    f"{self._safe_filename_slug(spec.prompt_text or Path(spec.mesh_path).stem)}"
                )
                sample_dir = samples_dir / sample_name
                sample_dir.mkdir(parents=True, exist_ok=True)

                record: dict[str, Any] = {
                    "sample_idx": sample_idx,
                    "epoch": epoch,
                    "global_step": global_step,
                    "prompt_text": spec.prompt_text,
                    "bbox_xyz": [float(value) for value in spec.bbox_xyz],
                    "reference_mesh_path": spec.mesh_path,
                    "sample_dir": str(sample_dir),
                }

                try:
                    bbox_tensor = torch.tensor(
                        [spec.bbox_xyz], dtype=torch.float32, device=self.device
                    )
                    shape_ids = self.sample_shape_ids(
                        prompt_text=[spec.prompt_text],
                        bbox_xyz=bbox_tensor,
                        num_diffusion_steps=num_steps,
                        top_p=top_p,
                        first_hitting=use_first_hitting,
                        first_hitting_tokens_per_step=tokens_per_step,
                    )
                    token_path = sample_dir / "shape_ids.pt"
                    torch.save(shape_ids.detach().cpu(), token_path)

                    with self.autocast_context():
                        mesh_v_f = self.decode_shape_ids(
                            shape_ids,
                            resolution_base=resolution_base,
                            chunk_size=chunk_size,
                        )
                    vertices, faces = mesh_v_f[0]
                    if vertices is None or faces is None:
                        raise RuntimeError("shape decode produced no valid surface")

                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                    mesh_path = sample_dir / "generated.obj"
                    mesh.export(mesh_path)

                    generated_samples += 1
                    record.update(
                        {
                            "status": "ok",
                            "generated_mesh_path": str(mesh_path),
                            "shape_ids_path": str(token_path),
                            "num_vertices": int(vertices.shape[0]),
                            "num_faces": int(faces.shape[0]),
                        }
                    )
                except Exception as error:
                    failed_samples += 1
                    logging.exception(
                        "Sample eval failed for step=%d sample_idx=%d mesh=%s",
                        global_step,
                        sample_idx,
                        spec.mesh_path,
                    )
                    record.update(
                        {
                            "status": "error",
                            "error": repr(error),
                        }
                    )

                _save_json(sample_dir / "metadata.json", record)
                _append_jsonl(sample_log_path, record)
                sample_records.append(record)
        finally:
            if self.ema is not None:
                self.ema.restore(self.gpt_model)
            if was_training:
                self.gpt_train_model.train()

        summary = {
            "event": "sample_eval",
            "epoch": epoch,
            "global_step": global_step,
            "requested_samples": requested_samples,
            "generated_samples": generated_samples,
            "failed_samples": failed_samples,
            "output_dir": str(samples_dir),
            "resolution_base": float(resolution_base),
            "chunk_size": int(chunk_size),
            "top_p": None if top_p is None else float(top_p),
            "num_diffusion_steps": num_steps,
            "first_hitting": use_first_hitting,
            "first_hitting_tokens_per_step": tokens_per_step,
            "duration_sec": time.perf_counter() - started_at,
            "samples": sample_records,
        }
        _save_json(samples_dir / "summary.json", summary)
        return summary

    def fit(
        self,
        dataloader: DataLoader,
        output_dir: str | Path,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        log_interval: int = 10,
        val_dataloader: Optional[DataLoader] = None,
        val_interval: int = 0,
        val_max_batches: Optional[int] = None,
        sample_eval_specs: Optional[list[SampleEvalSpec]] = None,
        sample_eval_interval: int = 0,
        sample_eval_resolution_base: float = 8.0,
        sample_eval_chunk_size: int = 100_000,
        sample_eval_top_p: Optional[float] = None,
        sample_eval_num_diffusion_steps: Optional[int] = None,
        sample_eval_first_hitting: Optional[bool] = None,
        sample_eval_first_hitting_tokens_per_step: Optional[int] = None,
        save_model_only_interval: int = 0,
        save_full_state_interval: int = 0,
        save_final_trainer_state: bool = False,
        resume_trainer_state: Optional[str] = None,
        tensorboard_dir: Optional[str] = None,
        enable_tensorboard: bool = True,
        tensorboard_flush_secs: int = 30,
        run_metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "checkpoints"
        logs_dir = output_dir / "logs"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        resolved_tensorboard_dir = (
            Path(tensorboard_dir).expanduser().resolve()
            if tensorboard_dir is not None
            else (logs_dir / "tensorboard").resolve()
        )

        metadata = {
            "config_path": str(self.config_path),
            "gpt_ckpt_path": str(self.gpt_ckpt_path) if self.gpt_ckpt_path else None,
            "shape_ckpt_path": str(self.shape_ckpt_path),
            "train_t_min": self.train_t_min,
            "train_t_max": self.train_t_max,
            "block_size": self.block_size,
            "generation_mode": getattr(self.gpt_model.cfg, "generation_mode", "ar"),
            "grad_accum_steps": self.grad_accum_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "ema_decay": self.ema.decay if self.ema is not None else 0.0,
            "model_dtype": self.model_dtype_name,
            "activation_checkpointing": bool(self.gpt_model.cfg.activation_checkpointing),
            "offload_shape_model_to_cpu": self.shape_model_offload_to_cpu,
            "save_full_state_interval": save_full_state_interval,
            "save_final_trainer_state": save_final_trainer_state,
            "tensorboard": {
                "enabled": bool(enable_tensorboard),
                "log_dir": str(resolved_tensorboard_dir),
                "flush_secs": int(max(tensorboard_flush_secs, 1)),
            },
        }
        if run_metadata is not None:
            metadata["run_metadata"] = run_metadata

        if self.is_main_process:
            OmegaConf.save(
                config=self.cfg,
                f=str(output_dir / "resolved_config.yaml"),
            )
            _save_json(output_dir / "training_run.json", metadata)
        self.distributed_barrier()

        if max_steps is None and epochs is None:
            raise ValueError(
                "Training requires max_steps or epochs. Prefer max_steps for step-driven training."
            )
        if max_steps is not None and int(max_steps) <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if epochs is not None and int(epochs) <= 0:
            raise ValueError(f"epochs must be positive when provided, got {epochs}")

        if max_steps is not None:
            total_optimizer_steps = max_steps
        else:
            assert epochs is not None
            steps_per_epoch = math.ceil(len(dataloader) / self.grad_accum_steps)
            total_optimizer_steps = max(1, steps_per_epoch * max(epochs, 1))
        self.scheduler = _build_cosine_scheduler(
            self.optimizer,
            total_steps=total_optimizer_steps,
            warmup_steps=self.warmup_steps,
            min_lr_ratio=self.min_lr_ratio,
        )

        resume_state = TrainerResumeState(
            global_step=0,
            epoch=1,
            resume_batch_idx=0,
            best_val_loss=None,
        )
        if resume_trainer_state is not None:
            resume_state = self.load_training_state(resume_trainer_state)
            logging.info(
                "Resumed training state from %s at epoch=%d global_step=%d resume_batch_idx=%d",
                resume_trainer_state,
                resume_state.epoch,
                resume_state.global_step,
                resume_state.resume_batch_idx,
            )

        train_log_path = logs_dir / "train.jsonl"
        val_log_path = logs_dir / "val.jsonl"
        sample_eval_log_path = logs_dir / "sample_eval.jsonl"
        if self.distributed_enabled:
            failure_log_path = logs_dir / f"failures_rank{self.distributed_rank:02d}.jsonl"
        else:
            failure_log_path = logs_dir / "failures.jsonl"
        global_step = resume_state.global_step
        best_val_loss = resume_state.best_val_loss
        last_epoch = max(1, resume_state.epoch)
        latest_model_path: Optional[Path] = None
        latest_model_step: Optional[int] = None
        self.optimizer.zero_grad(set_to_none=True)
        sample_eval_specs = sample_eval_specs or []
        tensorboard_writer: Optional[SummaryWriter] = None
        if enable_tensorboard and self.is_main_process:
            resolved_tensorboard_dir.mkdir(parents=True, exist_ok=True)
            tensorboard_writer = SummaryWriter(
                log_dir=str(resolved_tensorboard_dir),
                flush_secs=max(int(tensorboard_flush_secs), 1),
            )
        self.distributed_barrier()

        try:
            epoch = resume_state.epoch
            while True:
                if max_steps is None and epochs is not None and epoch > epochs:
                    break

                last_epoch = epoch
                skipped_batches = 0
                train_sampler = getattr(dataloader, "sampler", None)
                if hasattr(train_sampler, "set_epoch"):
                    train_sampler.set_epoch(epoch)

                progress = tqdm(
                    enumerate(dataloader),
                    total=len(dataloader),
                    desc=f"epoch {epoch}",
                    leave=False,
                    disable=not self.is_main_process,
                )
                last_batch_end = time.perf_counter()
                accum_counter = 0
                accum_loss_weighted_sum = 0.0
                accum_denoise_tokens = 0.0
                accum_total_tokens = 0.0
                accum_batch_size = 0.0
                step_data_time = 0.0
                optimizer_step_start: Optional[float] = None
                step_start_batch_idx = 0

                for batch_idx, batch in progress:
                    if max_steps is not None and global_step >= max_steps:
                        break
                    if epoch == resume_state.epoch and batch_idx < resume_state.resume_batch_idx:
                        continue
                    if batch is None:
                        skipped_batches += 1
                        continue

                    data_time = time.perf_counter() - last_batch_end
                    if accum_counter == 0:
                        accum_loss_weighted_sum = 0.0
                        accum_denoise_tokens = 0.0
                        accum_total_tokens = 0.0
                        accum_batch_size = 0.0
                        step_data_time = 0.0
                        optimizer_step_start = time.perf_counter()
                        step_start_batch_idx = batch_idx
                        self._reset_cuda_peak_memory_stats()
                    step_data_time += data_time

                    grad_norm_value = None
                    is_last_batch = batch_idx + 1 == len(dataloader)
                    should_step = accum_counter + 1 >= self.grad_accum_steps or is_last_batch
                    sync_context = (
                        self.gpt_train_model.no_sync()
                        if self.distributed_enabled and not should_step
                        else nullcontext()
                    )

                    try:
                        self.gpt_train_model.train()
                        with sync_context:
                            loss, metrics = self.compute_loss(batch)
                            if not bool(torch.isfinite(loss.detach()).item()):
                                raise FloatingPointError(
                                    f"Non-finite loss detected: {float(loss.detach().item())}"
                                )

                            loss_for_backward = loss / self.grad_accum_steps
                            if self.scaler.is_enabled():
                                self.scaler.scale(loss_for_backward).backward()
                            else:
                                loss_for_backward.backward()

                        accum_counter += 1
                        accum_loss_weighted_sum += (
                            float(loss.detach().item()) * float(metrics["denoise_tokens"])
                        )
                        accum_denoise_tokens += float(metrics["denoise_tokens"])
                        accum_total_tokens += float(metrics["total_tokens"])
                        accum_batch_size += float(metrics["batch_size"])

                        if not should_step:
                            last_batch_end = time.perf_counter()
                            continue

                        microbatches_in_step = accum_counter
                        grad_norm = None
                        if self.grad_clip_norm > 0:
                            if self.scaler.is_enabled():
                                self.scaler.unscale_(self.optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.gpt_model.parameters(), self.grad_clip_norm
                            )
                            grad_norm_value = float(grad_norm)
                            if not math.isfinite(grad_norm_value):
                                raise FloatingPointError(
                                    f"Non-finite grad_norm detected: {grad_norm_value}"
                                )

                        if self.scaler.is_enabled():
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        accum_counter = 0
                        global_step += 1
                        if self.scheduler is not None:
                            self.scheduler.step()
                        if self.ema is not None:
                            self.ema.update(self.gpt_model)
                    except Exception as error:
                        self.optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        latest_model_path, latest_model_error = self._try_save_latest_model_checkpoint(
                            checkpoint_dir
                        )
                        if isinstance(error, FloatingPointError):
                            failure_type = "non_finite"
                        elif _is_out_of_memory_error(error):
                            failure_type = "out_of_memory"
                        else:
                            failure_type = "exception"
                        failure_event = {
                            "event": "train_failure",
                            "failure_type": failure_type,
                            "rank": self.distributed_rank,
                            "world_size": self.distributed_world_size,
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "global_step": global_step,
                            "data_time": data_time,
                            "step_time": 0.0
                            if optimizer_step_start is None
                            else time.perf_counter() - optimizer_step_start,
                            "error": repr(error),
                            "batch": _batch_debug_summary(batch),
                            "latest_model_checkpoint": latest_model_path,
                            "latest_model_checkpoint_error": latest_model_error,
                        }
                        _append_jsonl(failure_log_path, failure_event)
                        logging.exception(
                            "Training step failed at epoch=%d batch_idx=%d global_step=%d",
                            epoch,
                            batch_idx,
                            global_step,
                        )
                        raise

                    if optimizer_step_start is None:
                        optimizer_step_start = time.perf_counter()
                    step_time = time.perf_counter() - optimizer_step_start
                    cuda_peak_memory = self._cuda_peak_memory_stats_mb()
                    step_metrics = self._reduce_step_metrics(
                        loss_weighted_sum=accum_loss_weighted_sum,
                        denoise_tokens=accum_denoise_tokens,
                        total_tokens=accum_total_tokens,
                        batch_size=accum_batch_size,
                    )
                    if grad_norm_value is not None and self.distributed_enabled:
                        grad_norm_value = (
                            self._distributed_sum(grad_norm_value)[0]
                            / float(self.distributed_world_size)
                        )

                    train_event = {
                        "event": "train",
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "step_start_batch_idx": step_start_batch_idx,
                        "microbatches": microbatches_in_step,
                        "global_step": global_step,
                        "loss": step_metrics["loss"],
                        "mask_ratio": step_metrics["mask_ratio"],
                        "denoise_tokens": step_metrics["denoise_tokens"],
                        "global_batch_size": step_metrics["batch_size"],
                        "lr": _current_lr(self.optimizer),
                        "grad_norm": grad_norm_value,
                        "step_time": step_time,
                        "data_time": step_data_time,
                        "skipped_batches": skipped_batches,
                        "max_memory_allocated_mb": cuda_peak_memory["max_memory_allocated_mb"],
                        "max_memory_reserved_mb": cuda_peak_memory["max_memory_reserved_mb"],
                    }

                    if self.is_main_process:
                        progress.set_postfix(
                            loss=f"{step_metrics['loss']:.4f}",
                            mask=f"{step_metrics['mask_ratio']:.3f}",
                            lr=f"{train_event['lr']:.2e}",
                        )
                        if tensorboard_writer is not None:
                            tensorboard_writer.add_scalar(
                                "train/loss", train_event["loss"], global_step
                            )
                            tensorboard_writer.add_scalar(
                                "train/mask_ratio", train_event["mask_ratio"], global_step
                            )
                            tensorboard_writer.add_scalar(
                                "train/denoise_tokens",
                                train_event["denoise_tokens"],
                                global_step,
                            )
                            tensorboard_writer.add_scalar(
                                "train/lr", train_event["lr"], global_step
                            )
                            tensorboard_writer.add_scalar(
                                "train/global_batch_size",
                                train_event["global_batch_size"],
                                global_step,
                            )
                            tensorboard_writer.add_scalar(
                                "train/step_time_sec",
                                train_event["step_time"],
                                global_step,
                            )
                            tensorboard_writer.add_scalar(
                                "train/data_time_sec",
                                train_event["data_time"],
                                global_step,
                            )
                            tensorboard_writer.add_scalar(
                                "train/max_memory_allocated_mb",
                                train_event["max_memory_allocated_mb"],
                                global_step,
                            )
                            tensorboard_writer.add_scalar(
                                "train/max_memory_reserved_mb",
                                train_event["max_memory_reserved_mb"],
                                global_step,
                            )
                            if grad_norm_value is not None:
                                tensorboard_writer.add_scalar(
                                    "train/grad_norm", grad_norm_value, global_step
                                )

                        if global_step % max(log_interval, 1) == 0:
                            logging.info(
                                "step=%d epoch=%d loss=%.4f mask_ratio=%.3f denoise_tokens=%d lr=%.2e grad_norm=%s global_batch=%d peak_alloc=%.1fMB peak_reserved=%.1fMB",
                                global_step,
                                epoch,
                                step_metrics["loss"],
                                step_metrics["mask_ratio"],
                                int(step_metrics["denoise_tokens"]),
                                train_event["lr"],
                                "n/a"
                                if grad_norm_value is None
                                else f"{grad_norm_value:.4f}",
                                int(step_metrics["batch_size"]),
                                train_event["max_memory_allocated_mb"],
                                train_event["max_memory_reserved_mb"],
                            )
                            _append_jsonl(train_log_path, train_event)

                    should_update_best = False
                    if (
                        val_dataloader is not None
                        and val_interval > 0
                        and global_step % val_interval == 0
                    ):
                        val_metrics = self.evaluate(val_dataloader, max_batches=val_max_batches)
                        if self.is_main_process:
                            val_event = {
                                "event": "val",
                                "epoch": epoch,
                                "global_step": global_step,
                                "loss": val_metrics["loss"],
                                "mask_ratio": val_metrics["mask_ratio"],
                                "denoise_tokens": val_metrics["denoise_tokens"],
                                "num_batches": val_metrics["num_batches"],
                                "skipped_batches": val_metrics["skipped_batches"],
                            }
                            _append_jsonl(val_log_path, val_event)
                            logging.info(
                                "val step=%d epoch=%d loss=%.4f mask_ratio=%.3f",
                                global_step,
                                epoch,
                                val_metrics["loss"],
                                val_metrics["mask_ratio"],
                            )
                            if tensorboard_writer is not None:
                                tensorboard_writer.add_scalar(
                                    "val/loss", val_metrics["loss"], global_step
                                )
                                tensorboard_writer.add_scalar(
                                    "val/mask_ratio",
                                    val_metrics["mask_ratio"],
                                    global_step,
                                )
                                tensorboard_writer.add_scalar(
                                    "val/denoise_tokens",
                                    val_metrics["denoise_tokens"],
                                    global_step,
                                )
                        if not math.isnan(val_metrics["loss"]) and (
                            best_val_loss is None or val_metrics["loss"] < best_val_loss
                        ):
                            best_val_loss = val_metrics["loss"]
                            should_update_best = True

                    if (
                        sample_eval_specs
                        and sample_eval_interval > 0
                        and global_step % sample_eval_interval == 0
                    ):
                        if self.is_main_process:
                            sample_eval_event = self.run_sample_generation_eval(
                                sample_eval_specs=sample_eval_specs,
                                output_dir=output_dir,
                                global_step=global_step,
                                epoch=epoch,
                                resolution_base=sample_eval_resolution_base,
                                chunk_size=sample_eval_chunk_size,
                                top_p=sample_eval_top_p,
                                num_diffusion_steps=sample_eval_num_diffusion_steps,
                                first_hitting=sample_eval_first_hitting,
                                first_hitting_tokens_per_step=sample_eval_first_hitting_tokens_per_step,
                            )
                            _append_jsonl(sample_eval_log_path, sample_eval_event)
                            logging.info(
                                "sample eval step=%d epoch=%d generated=%d failed=%d dir=%s",
                                global_step,
                                epoch,
                                sample_eval_event["generated_samples"],
                                sample_eval_event["failed_samples"],
                                sample_eval_event["output_dir"],
                            )
                            if tensorboard_writer is not None:
                                tensorboard_writer.add_scalar(
                                    "sample_eval/generated_samples",
                                    sample_eval_event["generated_samples"],
                                    global_step,
                                )
                                tensorboard_writer.add_scalar(
                                    "sample_eval/failed_samples",
                                    sample_eval_event["failed_samples"],
                                    global_step,
                                )
                                tensorboard_writer.add_scalar(
                                    "sample_eval/duration_sec",
                                    sample_eval_event["duration_sec"],
                                    global_step,
                                )
                        self.distributed_barrier()

                    if (
                        save_model_only_interval > 0
                        and global_step % save_model_only_interval == 0
                    ):
                        if self.is_main_process:
                            self.save_model_checkpoint(
                                checkpoint_dir=checkpoint_dir,
                                tag=f"step_{global_step:08d}",
                            )
                        self.distributed_barrier()

                    if (
                        save_full_state_interval > 0
                        and global_step % save_full_state_interval == 0
                    ):
                        if self.is_main_process:
                            latest_model_path = self.save_model_checkpoint(
                                checkpoint_dir=checkpoint_dir,
                                tag="latest",
                            )
                            latest_model_step = global_step
                            self.save_training_state(
                                checkpoint_dir=checkpoint_dir,
                                tag="latest",
                                global_step=global_step,
                                epoch=epoch,
                                resume_batch_idx=batch_idx + 1,
                                best_val_loss=best_val_loss,
                                model_checkpoint_path=latest_model_path,
                            )
                        self.distributed_barrier()

                    if should_update_best:
                        if self.is_main_process:
                            if (
                                self.ema is None
                                and latest_model_path is not None
                                and latest_model_step == global_step
                            ):
                                _materialize_file_alias(
                                    latest_model_path,
                                    self.model_checkpoint_path(checkpoint_dir, "best"),
                                )
                            else:
                                self.save_model_checkpoint(
                                    checkpoint_dir=checkpoint_dir,
                                    tag="best",
                                    use_ema=self.ema is not None,
                                )
                        self.distributed_barrier()

                    last_batch_end = time.perf_counter()
                    if max_steps is not None and global_step >= max_steps:
                        break

                if val_dataloader is not None and global_step > 0 and val_interval <= 0:
                    val_metrics = self.evaluate(val_dataloader, max_batches=val_max_batches)
                    if self.is_main_process:
                        val_event = {
                            "event": "val",
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": val_metrics["loss"],
                            "mask_ratio": val_metrics["mask_ratio"],
                            "denoise_tokens": val_metrics["denoise_tokens"],
                            "num_batches": val_metrics["num_batches"],
                            "skipped_batches": val_metrics["skipped_batches"],
                        }
                        _append_jsonl(val_log_path, val_event)
                        logging.info(
                            "val epoch=%d step=%d loss=%.4f mask_ratio=%.3f",
                            epoch,
                            global_step,
                            val_metrics["loss"],
                            val_metrics["mask_ratio"],
                        )
                        if tensorboard_writer is not None:
                            tensorboard_writer.add_scalar(
                                "val/loss", val_metrics["loss"], global_step
                            )
                            tensorboard_writer.add_scalar(
                                "val/mask_ratio", val_metrics["mask_ratio"], global_step
                            )
                            tensorboard_writer.add_scalar(
                                "val/denoise_tokens",
                                val_metrics["denoise_tokens"],
                                global_step,
                            )
                    if not math.isnan(val_metrics["loss"]) and (
                        best_val_loss is None or val_metrics["loss"] < best_val_loss
                    ):
                        best_val_loss = val_metrics["loss"]
                        if self.is_main_process:
                            if (
                                self.ema is None
                                and latest_model_path is not None
                                and latest_model_step == global_step
                            ):
                                _materialize_file_alias(
                                    latest_model_path,
                                    self.model_checkpoint_path(checkpoint_dir, "best"),
                                )
                            else:
                                self.save_model_checkpoint(
                                    checkpoint_dir=checkpoint_dir,
                                    tag="best",
                                    use_ema=self.ema is not None,
                                )
                        self.distributed_barrier()

                resume_state = TrainerResumeState(
                    global_step=global_step,
                    epoch=epoch + 1,
                    resume_batch_idx=0,
                    best_val_loss=best_val_loss,
                )
                if max_steps is not None and global_step >= max_steps:
                    break
                epoch += 1

            if val_dataloader is not None and global_step > 0 and (
                val_interval > 0 and global_step % val_interval != 0
            ):
                val_metrics = self.evaluate(val_dataloader, max_batches=val_max_batches)
                if self.is_main_process:
                    val_event = {
                        "event": "val",
                        "epoch": last_epoch,
                        "global_step": global_step,
                        "loss": val_metrics["loss"],
                        "mask_ratio": val_metrics["mask_ratio"],
                        "denoise_tokens": val_metrics["denoise_tokens"],
                        "num_batches": val_metrics["num_batches"],
                        "skipped_batches": val_metrics["skipped_batches"],
                    }
                    _append_jsonl(val_log_path, val_event)
                    if tensorboard_writer is not None:
                        tensorboard_writer.add_scalar(
                            "val/loss", val_metrics["loss"], global_step
                        )
                        tensorboard_writer.add_scalar(
                            "val/mask_ratio", val_metrics["mask_ratio"], global_step
                        )
                        tensorboard_writer.add_scalar(
                            "val/denoise_tokens",
                            val_metrics["denoise_tokens"],
                            global_step,
                        )
                if not math.isnan(val_metrics["loss"]) and (
                    best_val_loss is None or val_metrics["loss"] < best_val_loss
                ):
                    best_val_loss = val_metrics["loss"]
                    if self.is_main_process:
                        if (
                            self.ema is None
                            and latest_model_path is not None
                            and latest_model_step == global_step
                        ):
                            _materialize_file_alias(
                                latest_model_path,
                                self.model_checkpoint_path(checkpoint_dir, "best"),
                            )
                        else:
                            self.save_model_checkpoint(
                                checkpoint_dir=checkpoint_dir,
                                tag="best",
                                use_ema=self.ema is not None,
                            )
                    self.distributed_barrier()

            final_model_path = self.model_checkpoint_path(checkpoint_dir, "final")
            if self.is_main_process:
                if latest_model_path is not None and latest_model_step == global_step:
                    final_model_path = _materialize_file_alias(
                        latest_model_path,
                        self.model_checkpoint_path(checkpoint_dir, "final"),
                    )
                else:
                    final_model_path = self.save_model_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        tag="final",
                    )
                if save_full_state_interval > 0 or save_final_trainer_state:
                    self.save_training_state(
                        checkpoint_dir=checkpoint_dir,
                        tag="latest",
                        global_step=global_step,
                        epoch=last_epoch,
                        resume_batch_idx=0,
                        best_val_loss=best_val_loss,
                        model_checkpoint_path=final_model_path,
                    )
                if save_final_trainer_state:
                    self.save_training_state(
                        checkpoint_dir=checkpoint_dir,
                        tag="final",
                        global_step=global_step,
                        epoch=last_epoch,
                        resume_batch_idx=0,
                        best_val_loss=best_val_loss,
                        model_checkpoint_path=final_model_path,
                    )
            self.distributed_barrier()
            return final_model_path
        finally:
            if tensorboard_writer is not None:
                tensorboard_writer.flush()
                tensorboard_writer.close()
