# Cube Block Diffusion Implementation Notes

## 1. Background and Goal

This document records the full set of code changes made to convert Cube's text-to-shape GPT from a purely autoregressive (AR) decoder into a Block Diffusion style decoder while keeping the `DualStreamRoformer` backbone intact.

The main objective was:

- Keep the existing `DualStreamRoformer` architecture as the backbone.
- Preserve the original AR path for compatibility.
- Add a Block Diffusion generation and training mode that increases parallelism within each shape-token block.
- Reuse Cube's existing text tokenizer / text encoder path.
- Reuse Cube's existing shape autoencoder to produce shape tokens from local Objaverse meshes.
- Add a minimal local training pipeline that starts from mesh + text pairs rather than pre-tokenized training data.

The implementation follows the user-specified design rather than aiming for a line-by-line reproduction of the original Block Diffusion paper codebase.

## 2. Core Modeling Change

### 2.1 Original AR semantics

In the original Cube text-to-shape GPT:

- The hidden state at position `i` is used to predict token `i + 1`.
- Each position only sees previous positions.
- Decoding is strictly sequential.

### 2.2 New Block Diffusion semantics

In the new Block Diffusion path:

- A block of shape tokens is denoised as a unit.
- The hidden state at position `i` is interpreted as the prediction for position `i` itself, not for `i + 1`.
- Tokens inside the current block attend bidirectionally to each other.
- Blocks can attend only to earlier blocks.
- This gives:
  - block-to-block: causal / one-way
  - within-block: bidirectional

This is the key change that enables parallel denoising inside a block.

## 3. High-Level Architecture Decisions

- The `DualStreamRoformer` module remains the core transformer backbone.
- The AR code path was preserved to avoid breaking the original inference flow.
- A separate Block Diffusion forward path was added instead of replacing the AR path.
- The default Block Diffusion mask token uses `padding_id` when `mask_token_id < 0`.
- This was chosen to avoid immediately adding a brand-new vocabulary token and breaking old GPT checkpoint loading behavior.
- The new training loss is standard cross-entropy on masked noisy positions only.
- The harder BD3-LM SUBS parameterization was intentionally not implemented at this stage.

## 4. Files Changed

### 4.1 Existing files modified

- `cube3d/model/gpt/dual_stream_roformer.py`
- `cube3d/inference/engine.py`
- `cube3d/inference/logits_postprocesses.py`
- `cube3d/generate.py`
- `pyproject.toml`

### 4.2 New files added

- `cube3d/model/gpt/block_diffusion_utils.py`
- `cube3d/training/data.py`
- `cube3d/training/block_diffusion.py`
- `cube3d/training/__init__.py`
- `cube3d/train_block_diffusion.py`
- `cube3d/configs/open_model_block_diffusion.yaml`
- `BLOCK_DIFFUSION_IMPLEMENTATION.md`

## 5. Detailed Code Changes

### 5.1 `cube3d/model/gpt/dual_stream_roformer.py`

This file is the backbone-level change.

New config fields were added:

- `generation_mode: str = "ar"`
- `block_size: int = 8`
- `mask_token_id: int = -1`
- `use_single_blocks_in_diffusion: bool = False`

New model behavior:

- Added `shape_mask_id`.
- If `mask_token_id < 0`, `shape_mask_id` falls back to `padding_id`.
- The forward pass was refactored into:
  - `_compute_shape_position_ids`
  - `_compute_rotary_embeddings`
  - `_run_blocks`
  - `forward_ar`
  - `forward_block_diffusion`
- `forward()` now dispatches to the Block Diffusion path when `attn_mask` or `shape_position_ids` is provided.

Block Diffusion-specific details:

- `forward_block_diffusion()` accepts a custom full attention mask.
- It also accepts duplicated shape position ids so `[clean_shape | noisy_shape]` can share the same RoPE positions.
- `use_single_blocks` is configurable and defaults to `cfg.use_single_blocks_in_diffusion`.

AR compatibility:

- `forward_ar()` preserves the original causal generation logic.
- Existing AR inference code remains valid.

### 5.2 `cube3d/model/gpt/block_diffusion_utils.py`

This file centralizes mask and schedule helpers used by both inference and training.

Implemented helpers:

- `duplicate_shape_position_ids`
- `build_training_shape_attention_mask`
- `build_inference_shape_attention_mask`
- `wrap_shape_attention_with_condition_prefix`
- `sample_block_timesteps`
- `mask_shape_tokens`
- `linear_noise_schedule`
- `mask_keep_probability`
- `sample_first_hitting_positions`
- `build_block_denoise_loss_mask`

Important mask definitions:

- Training mask uses the shape layout `[clean_shape | noisy_shape]`.
- Clean block `k` sees clean blocks `<= k`.
- Noisy block `k` sees clean blocks `< k`.
- Noisy block `k` sees itself bidirectionally.
- Noisy block `k` does not see other noisy blocks.
- Noisy block `k` does not see clean block `k` or any later clean block.

This is the mechanism used to prevent answer leakage while preserving the desired block-unique denoising context.

### 5.3 `cube3d/inference/logits_postprocesses.py`

A bug was fixed here:

- `process_logits(..., top_p)` previously ignored the incoming `top_p` value and effectively hardcoded `0.9`.
- It now respects the caller-provided `top_p`.
- Sampling was generalized so logits with arbitrary leading dimensions like `[B, L, V]` can be sampled correctly.

This matters because the Block Diffusion path samples an entire block of logits at once rather than a single next-token distribution.

### 5.4 `cube3d/inference/engine.py`

This file now supports both AR and Block Diffusion inference.

New engine state:

- `self.generation_mode`
- `self.block_size`
- `self.mask_token_id`

New helper methods:

- `prepare_conditions(...)`
- `encode_shape_tokens(...)`
- `run_block_diffusion_gpt(...)`

`run_block_diffusion_gpt(...)` behavior:

- Operates block by block over the shape token sequence.
- The current block is initialized as all mask tokens.
- The model input for each denoise step is `[context | x_t_block]`.
- Earlier finalized blocks are used as context.
- The current block is predicted in parallel.
- Only masked positions are updated.
- Logits are restricted to the shape codebook slice `[:num_codes]`.
- `top_p` sampling is supported.

Two denoising strategies were implemented:

- `first_hitting=False`
- `first_hitting=True`

For `first_hitting=False`:

- A simple linear schedule is created from `t` to `s`.
- `mask_keep_probability(t, s) = s / t` is used.
- For each masked token, a Bernoulli decision determines whether the token remains masked or gets replaced.
- This is applied independently per token.
- A final cleanup pass fills any remaining masks.

For `first_hitting=True`:

- The implementation follows the requested "respect the paper-style first-hitting" behavior.
- Each step randomly reveals a small number of masked positions.
- The default is 1 token per step, but this is configurable.

Important current limitation:

- Block Diffusion inference does not use KV-cache optimization.
- It runs a full forward pass at each denoise step.

Conditioning:

- Bounding-box conditioning continues to work through `bbox_proj`.
- Classifier-free guidance remains only in the AR path.
- The Block Diffusion path currently does not implement CFG.

`EngineFast`:

- Explicitly remains AR-only.
- It now asserts that `generation_mode == "ar"`.

### 5.5 `cube3d/generate.py`

The CLI generation script was extended so Block Diffusion inference can be controlled from the command line.

New arguments:

- `--num-diffusion-steps`
- `--first-hitting`
- `--first-hitting-tokens-per-step`

These are passed through `engine.t2s(...)`.

### 5.6 `cube3d/training/data.py`

This is the new local Objaverse-style training dataset loader.

Supported data discovery modes:

- Manifest-based loading
- Recursive filesystem scan

Supported manifest formats:

- `.json`
- `.jsonl`
- `.csv`
- `.tsv`

Supported mesh suffixes:

- `.obj`
- `.ply`
- `.glb`
- `.gltf`
- `.stl`
- `.off`

Supported text sources:

- Manifest keys such as `text`, `caption`, `prompt`, `description`, `name`, `title`
- Sidecar `.txt`
- Sidecar `.json`
- Fallback to filename stem

Per-sample preprocessing:

- Load and clean mesh
- Rescale mesh to the expected Cube unit cube
- Sample a surface point cloud with normals
- Compute normalized bounding box
- Return:
  - `prompt_text`
  - `point_cloud`
  - `bbox_xyz`
  - `mesh_path`

Caching:

- Optional cache directory support was added.
- Cache filenames use a hash of the mesh path to reduce collisions.

Dependency handling:

- `trimesh` import was made lazy so importing training utilities does not fail unless mesh loading is actually used.

### 5.7 `cube3d/training/block_diffusion.py`

This file implements the new minimal trainer.

Main class:

- `CubeBlockDiffusionTrainer`

Key responsibilities:

- Load config
- Load GPT model
- Optionally load a GPT initialization checkpoint
- Load shape autoencoder checkpoint
- Load CLIP tokenizer and text encoder
- Freeze the shape autoencoder
- Freeze the CLIP text encoder
- Copy the VQ codebook into GPT token embeddings exactly like the inference engine
- Encode prompt text into condition embeddings
- Encode point clouds into shape token ids
- Build Block Diffusion noisy inputs
- Compute masked denoising loss
- Optimize GPT parameters
- Save checkpoints

Checkpoint loading:

- Supports `.safetensors` if `safetensors` is installed.
- Also supports plain `torch.load(...)` checkpoints.
- For non-safetensors checkpoints, it tries keys like:
  - `gpt_model`
  - `shape_model`
  - `model`
  - `state_dict`

Training input construction:

- Clean shape tokens are encoded from the local mesh using the shape autoencoder.
- Condition tokens come from Cube's existing CLIP path plus optional bbox.
- Only the shape side is noised.
- The final shape-side training sequence is:
  - `[clean_shape | noisy_shape]`

Timesteps and noising:

- A block-level `t` is sampled uniformly in `[train_t_min, train_t_max]`.
- The same `t` is repeated across tokens in the same block.
- Default range is `[0.45, 0.95]`.
- Tokens are masked according to this block timestep.

Position ids:

- `duplicate_shape_position_ids(...)` is used so clean and noisy halves share the same shape positions.

Loss:

- Only the noisy half is supervised.
- Only locations still equal to `mask_token_id` contribute to loss.
- Cross-entropy is averaged over valid masked denoise positions.
- Unmasked noisy positions are excluded from the loss.

Attention mask:

- Training uses the custom cross-sequence mask described above.
- The full attention mask passed to the transformer is:
  - `[cond | clean_shape | noisy_shape]`

Checkpoint saving:

- GPT weights are saved as `.safetensors` when possible.
- A separate trainer state is saved as `.pt`.

Current scope:

- Single-process, minimal PyTorch training loop
- No distributed training support
- No resume-from-checkpoint command line flow yet
- No evaluation pipeline yet

### 5.8 `cube3d/training/__init__.py`

This file now:

- Exports dataset helpers directly
- Lazily imports `CubeBlockDiffusionTrainer`

Reason:

- Importing `cube3d.training` should not eagerly require the full trainer dependency stack if the caller only wants dataset utilities.

### 5.9 `cube3d/train_block_diffusion.py`

This is the new training entrypoint.

It provides a minimal CLI for:

- dataset root
- optional manifest path
- config path
- optional GPT init checkpoint
- shape checkpoint
- output directory
- batch size
- num workers
- epochs
- max steps
- learning rate
- weight decay
- grad clipping
- save interval
- log interval
- point sampling count
- cache directory
- timestep range
- seed
- device override

It constructs:

- `ObjaverseDataset`
- `DataLoader`
- `CubeBlockDiffusionTrainer`

Then it calls:

- `trainer.fit(...)`

### 5.10 `cube3d/configs/open_model_block_diffusion.yaml`

This new config is based on the existing open model configuration and sets the model to Block Diffusion mode.

Important values:

- `generation_mode: block_diffusion`
- `block_size: 8`
- `mask_token_id: -1`
- `use_single_blocks_in_diffusion: false`

It also records Block Diffusion defaults:

- `train_t_min: 0.45`
- `train_t_max: 0.95`
- `num_diffusion_steps: 32`
- `first_hitting: false`
- `first_hitting_tokens_per_step: 1`

### 5.11 `pyproject.toml`

Dependency declaration was updated to include:

- `safetensors`

This is necessary because:

- The original inference utilities already load safetensor checkpoints.
- The new trainer can also save GPT checkpoints as `.safetensors`.

## 6. Training Logic Summary

The final training logic implemented in code is:

- Read local Objaverse mesh + text data.
- Use Cube's CLIP tokenizer and text encoder path to create text conditions.
- Use Cube's shape autoencoder to encode each mesh into shape token ids.
- Compute bbox from the mesh and inject it into the condition stream if the GPT config uses bbox.
- Build a clean shape sequence `x`.
- Sample a block-level mask ratio `t ~ Uniform(0.45, 0.95)` by default.
- Apply masking only on the shape side.
- Build `[clean_shape | noisy_shape]`.
- Build the special cross-sequence attention mask so the noisy half only sees:
  - clean blocks before the current block
  - itself bidirectionally
- Run `forward_block_diffusion(...)`.
- Slice logits to the noisy half.
- Compute cross-entropy only on positions that are still mask tokens.
- Average across valid denoising positions.

This matches the intended "clean sequence as context, noisy sequence as target" design.

## 7. Inference Logic Summary

The final inference logic implemented in code is:

- Encode the text condition.
- Optionally add bbox conditioning.
- Initialize shape tokens as fully masked.
- Denoise block by block.
- For the current block, build `[context | x_t_block]`.
- Use an attention mask where:
  - context is causal
  - the current block sees all context
  - the current block is fully bidirectional internally
- Predict logits for the whole current block.
- Only masked positions are eligible for replacement.
- Use `top_p` sampling if requested.

Two denoising strategies are available:

- `first_hitting=False`
- `first_hitting=True`

`first_hitting=False`:

- More parallel.
- Uses keep-mask probability to decide which masked positions remain masked.

`first_hitting=True`:

- More conservative.
- Randomly reveals a small number of masked tokens per step until the block is complete.

## 8. Compatibility and Safety Notes

- AR generation was not removed.
- Old AR inference entrypoints still exist.
- `EngineFast` is still available for AR only.
- The default Block Diffusion mask token reuses `padding_id` when no dedicated mask token is configured.
- This improves checkpoint compatibility but should be kept in mind when changing vocab semantics later.
- The trainer copies the VQ codebook into GPT embeddings after loading the shape autoencoder, matching the inference engine behavior.

## 9. Validation Performed

The following checks were run after implementation:

- `python -m compileall /root/cube/cube3d`
- A synthetic smoke test for `DualStreamRoformer.forward_ar(...)`
- A synthetic smoke test for `DualStreamRoformer.forward_block_diffusion(...)`
- Import check for training dataset utilities

Validation that was not run in this environment:

- Full end-to-end training on real Objaverse data
- Real checkpoint loading with the trainer
- Full Block Diffusion inference with production checkpoints

The main reason is that the sandbox runtime used during implementation did not have the complete dependency stack installed for a full training run.

## 10. Known Limitations

- Block Diffusion inference currently uses full forward passes and does not have a KV-cache optimized implementation.
- Classifier-free guidance is currently only preserved in the AR path.
- The trainer is intentionally minimal and does not yet include:
  - distributed training
  - AMP gradient scaling management beyond PyTorch autocast
  - checkpoint resume CLI
  - validation / evaluation loop
  - data quality filtering
- The current loss is plain masked cross-entropy, not a specialized SUBS parameterization.
- The default mask token is `padding_id` unless a dedicated mask token id is introduced later.

## 11. Suggested Next Steps

The most likely next engineering steps are:

- Add a dedicated mask token if checkpoint migration is acceptable.
- Add classifier-free guidance support for Block Diffusion inference.
- Add resume-from-checkpoint support to the trainer CLI.
- Add validation metrics and sample generation during training.
- Explore KV-cache or partial-cache acceleration for Block Diffusion decoding.
- Revisit whether `single_blocks` should participate in the Block Diffusion path after more experiments.
- If needed, replace the current CE head with a stricter Block Diffusion / BD3-LM style output parameterization.

## 12. Minimal Usage Example

Example training command:

```bash
python cube3d/train_block_diffusion.py \
  --data-root /path/to/objaverse \
  --shape-ckpt-path /path/to/shape_tokenizer.safetensors \
  --gpt-ckpt-path /path/to/shape_gpt.safetensors \
  --output-dir /path/to/block_diffusion_runs
```

Example generation command in Block Diffusion mode:

```bash
python cube3d/generate.py \
  --config-path cube3d/configs/open_model_block_diffusion.yaml \
  --gpt-ckpt-path /path/to/gpt_checkpoint.safetensors \
  --shape-ckpt-path /path/to/shape_tokenizer.safetensors \
  --prompt "a wooden chair" \
  --num-diffusion-steps 32 \
  --top-p 0.95
```

## 13. Final Note

This implementation is intended as a strong working baseline for the next phase of model iteration. It preserves the original Cube AR path, adds the requested Block Diffusion behavior, and introduces a practical local training path based on mesh-caption data from Objaverse while keeping the `DualStreamRoformer` backbone unchanged.
