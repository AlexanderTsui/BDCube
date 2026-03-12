import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from cube3d.training import (
    CubeBlockDiffusionTrainer,
    ObjaverseDataset,
    collate_objaverse_batch,
)
from cube3d.training.block_diffusion import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Cube text-to-shape GPT with block diffusion."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of the local Objaverse-style dataset.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional manifest (.json/.jsonl/.csv/.tsv) describing mesh/text pairs.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="cube3d/configs/open_model_block_diffusion.yaml",
        help="Path to the model config.",
    )
    parser.add_argument(
        "--gpt-ckpt-path",
        type=str,
        default=None,
        help="Optional GPT initialization checkpoint.",
    )
    parser.add_argument(
        "--shape-ckpt-path",
        type=str,
        required=True,
        help="Checkpoint for Cube's shape autoencoder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory used for checkpoints and logs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on optimizer steps. Overrides full-epoch training when set.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Global gradient clipping norm. Use 0 to disable.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Save a checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log metrics every N optimizer steps.",
    )
    parser.add_argument(
        "--point-samples",
        type=int,
        default=8192,
        help="Number of surface points sampled per mesh.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for sampled point clouds and bbox data.",
    )
    parser.add_argument(
        "--train-t-min",
        type=float,
        default=None,
        help="Minimum per-block masking rate during training.",
    )
    parser.add_argument(
        "--train-t-max",
        type=float,
        default=None,
        help="Maximum per-block masking rate during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional explicit device, e.g. cuda, cpu, mps, cuda:0.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = select_device()
    logging.info("Using device: %s", device)

    dataset = ObjaverseDataset(
        root=args.data_root,
        manifest_path=args.manifest_path,
        point_samples=args.point_samples,
        cache_dir=args.cache_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_objaverse_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    trainer = CubeBlockDiffusionTrainer(
        config_path=args.config_path,
        gpt_ckpt_path=args.gpt_ckpt_path,
        shape_ckpt_path=args.shape_ckpt_path,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_t_min=args.train_t_min,
        train_t_max=args.train_t_max,
        grad_clip_norm=args.grad_clip_norm,
    )
    final_checkpoint = trainer.fit(
        dataloader=dataloader,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )
    logging.info("Training finished. Final checkpoint: %s", final_checkpoint)


if __name__ == "__main__":
    main()
