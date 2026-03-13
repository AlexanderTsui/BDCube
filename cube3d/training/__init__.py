from cube3d.training.data import (
    DatasetDiscoverySummary,
    ObjaverseDataset,
    SampleEvalSpec,
    collate_objaverse_batch,
    discover_objaverse_entries,
    prepare_sample_eval_specs,
    split_objaverse_entries,
)

__all__ = [
    "CubeBlockDiffusionTrainer",
    "DatasetDiscoverySummary",
    "ObjaverseDataset",
    "SampleEvalSpec",
    "collate_objaverse_batch",
    "discover_objaverse_entries",
    "prepare_sample_eval_specs",
    "split_objaverse_entries",
]


def __getattr__(name: str):
    if name == "CubeBlockDiffusionTrainer":
        from cube3d.training.block_diffusion import CubeBlockDiffusionTrainer

        return CubeBlockDiffusionTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
