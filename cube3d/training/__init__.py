from cube3d.training.data import ObjaverseDataset, collate_objaverse_batch

__all__ = [
    "CubeBlockDiffusionTrainer",
    "ObjaverseDataset",
    "collate_objaverse_batch",
]


def __getattr__(name: str):
    if name == "CubeBlockDiffusionTrainer":
        from cube3d.training.block_diffusion import CubeBlockDiffusionTrainer

        return CubeBlockDiffusionTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
