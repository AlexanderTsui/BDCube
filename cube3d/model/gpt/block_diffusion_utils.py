from __future__ import annotations

import math
from typing import Optional

import torch


def _num_blocks(num_tokens: int, block_size: int) -> int:
    return math.ceil(num_tokens / block_size)


def duplicate_shape_position_ids(
    batch_size: int,
    num_shape_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    position_ids = torch.arange(num_shape_tokens, device=device, dtype=torch.long)
    position_ids = torch.cat([position_ids, position_ids], dim=0)
    return position_ids.unsqueeze(0).expand(batch_size, -1)


def build_training_shape_attention_mask(
    num_shape_tokens: int,
    block_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build the shape-side attention mask for [clean_shape | noisy_shape].

    clean block k:
      - can attend to clean blocks <= k
      - cannot attend to noisy half

    noisy block k:
      - can attend to clean blocks < k
      - can attend bidirectionally within noisy block k
      - cannot attend to clean block k or future clean blocks
      - cannot attend to any other noisy block
    """
    total_shape_len = num_shape_tokens * 2
    mask = torch.zeros(
        (total_shape_len, total_shape_len), dtype=torch.bool, device=device
    )
    num_blocks = _num_blocks(num_shape_tokens, block_size)

    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, num_shape_tokens)

        clean_start = block_start
        clean_end = block_end
        noisy_start = num_shape_tokens + block_start
        noisy_end = num_shape_tokens + block_end

        # clean block k sees clean blocks <= k
        mask[clean_start:clean_end, :clean_end] = True

        # noisy block k sees clean blocks < k
        mask[noisy_start:noisy_end, :clean_start] = True
        # noisy block k sees itself bidirectionally
        mask[noisy_start:noisy_end, noisy_start:noisy_end] = True

    return mask


def build_inference_shape_attention_mask(
    context_len: int,
    block_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build the shape-side attention mask for [context_shape | x_t_block].

    - context is causal and fully visible to the current block
    - current block is bidirectional within the block
    - current block does not see future tokens because they are absent
    """
    total_shape_len = context_len + block_len
    mask = torch.zeros(
        (total_shape_len, total_shape_len), dtype=torch.bool, device=device
    )
    if context_len > 0:
        mask[:context_len, :context_len] = torch.tril(
            torch.ones(
                (context_len, context_len), dtype=torch.bool, device=device
            )
        )
        mask[context_len:, :context_len] = True
    mask[context_len:, context_len:] = True
    return mask


def wrap_shape_attention_with_condition_prefix(
    shape_mask: torch.Tensor,
    cond_len: int,
) -> torch.Tensor:
    """
    Expand a shape-side mask into a full [cond | shape] mask.

    cond rows only attend to cond columns.
    shape rows attend to all cond columns plus the shape-side mask.
    """
    total_len = cond_len + shape_mask.shape[0]
    full_mask = torch.zeros(
        (total_len, total_len), dtype=torch.bool, device=shape_mask.device
    )
    if cond_len > 0:
        full_mask[:cond_len, :cond_len] = True
        full_mask[cond_len:, :cond_len] = True
    full_mask[cond_len:, cond_len:] = shape_mask
    return full_mask


def sample_block_timesteps(
    batch_size: int,
    num_shape_tokens: int,
    block_size: int,
    t_min: float,
    t_max: float,
    device: torch.device,
) -> torch.Tensor:
    num_blocks = _num_blocks(num_shape_tokens, block_size)
    t = torch.rand((batch_size, num_blocks), device=device)
    t = t * (t_max - t_min) + t_min
    return t.repeat_interleave(block_size, dim=1)[:, :num_shape_tokens]


def mask_shape_tokens(
    shape_ids: torch.Tensor,
    block_timesteps: torch.Tensor,
    mask_token_id: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if valid_mask is None:
        valid_mask = torch.ones_like(shape_ids, dtype=torch.bool)
    move_mask = (torch.rand_like(block_timesteps) < block_timesteps) & valid_mask
    noisy_ids = torch.where(
        move_mask,
        torch.full_like(shape_ids, fill_value=mask_token_id),
        shape_ids,
    )
    return noisy_ids, move_mask


def linear_noise_schedule(step_idx: int, num_steps: int) -> float:
    if num_steps <= 0:
        return 0.0
    return max(0.0, 1.0 - step_idx / num_steps)


def mask_keep_probability(t: float, s: float) -> float:
    if t <= 0.0:
        return 0.0
    return max(0.0, min(1.0, s / t))


def sample_first_hitting_positions(
    mask_positions: torch.Tensor,
    num_tokens_per_step: int,
) -> torch.Tensor:
    """
    Returns a boolean mask with up to num_tokens_per_step True entries
    within the provided mask_positions tensor.
    """
    update_mask = torch.zeros_like(mask_positions, dtype=torch.bool)
    if num_tokens_per_step <= 0:
        return update_mask

    batch_size = mask_positions.shape[0]
    for batch_idx in range(batch_size):
        active = torch.nonzero(mask_positions[batch_idx], as_tuple=False).flatten()
        if active.numel() == 0:
            continue
        perm = torch.randperm(active.numel(), device=mask_positions.device)
        chosen = active[perm[: min(num_tokens_per_step, active.numel())]]
        update_mask[batch_idx, chosen] = True
    return update_mask


def build_block_denoise_loss_mask(
    noisy_shape_ids: torch.Tensor,
    mask_token_id: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    denoise_mask = noisy_shape_ids.eq(mask_token_id)
    if valid_mask is not None:
        denoise_mask = denoise_mask & valid_mask
    return denoise_mask
