import tempfile
import unittest
from pathlib import Path

import torch

from cube3d.training.data import (
    ObjaverseDataset,
    collate_objaverse_batch,
    discover_objaverse_entries,
    normalize_bbox,
    prepare_sample_eval_specs,
    split_objaverse_entries,
)


class TrainingDataTest(unittest.TestCase):
    def test_discover_entries_prefers_sidecar_text_and_reports_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_path = root / "sample.obj"
            mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            mesh_path.with_suffix(".txt").write_text("clean prompt")

            entries, summary = discover_objaverse_entries(root)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["text"], "clean prompt")
            self.assertEqual(summary.total_records, 1)
            self.assertEqual(summary.valid_entries, 1)
            self.assertEqual(summary.sidecar_text_entries, 1)
            self.assertEqual(summary.stem_text_entries, 0)

    def test_split_entries_is_deterministic(self) -> None:
        entries = [{"mesh_path": f"mesh_{idx}.obj", "text": f"text_{idx}"} for idx in range(10)]
        split_a = split_objaverse_entries(entries, val_ratio=0.2, seed=123)
        split_b = split_objaverse_entries(entries, val_ratio=0.2, seed=123)

        self.assertEqual(split_a, split_b)
        self.assertEqual(len(split_a[0]), 8)
        self.assertEqual(len(split_a[1]), 2)

    def test_collate_skips_invalid_samples(self) -> None:
        batch = collate_objaverse_batch(
            [
                None,
                {
                    "prompt_text": "hello",
                    "point_cloud": torch.zeros(4, 6),
                    "bbox_xyz": torch.tensor([1.0, 1.0, 1.0]),
                    "mesh_path": "mesh.obj",
                },
            ]
        )

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch["prompt_text"], ["hello"])
        self.assertEqual(tuple(batch["point_cloud"].shape), (1, 4, 6))
        self.assertEqual(tuple(batch["bbox_xyz"].shape), (1, 3))

    def test_collate_returns_none_when_every_sample_is_invalid(self) -> None:
        self.assertIsNone(collate_objaverse_batch([None, None]))

    def test_discover_entries_resolves_pair_path_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_path = root / "mesh.obj"
            pair_path = root / "mesh_pair.pt"
            manifest_path = root / "manifest.jsonl"
            mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            torch.save(
                {
                    "bbox_xyz": torch.tensor([1.0, 1.5, 1.2], dtype=torch.float32),
                    "text_input_ids": torch.ones(1, 77, dtype=torch.long),
                    "text_attention_mask": torch.ones(1, 77, dtype=torch.long),
                    "shape_ids": torch.arange(1024, dtype=torch.long).unsqueeze(0),
                },
                pair_path,
            )
            manifest_path.write_text(
                '{"mesh_path": "mesh.obj", "text": "prompt", "pair_path": "mesh_pair.pt"}\n'
            )

            entries, summary = discover_objaverse_entries(root, manifest_path=manifest_path)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["mesh_path"], mesh_path.resolve())
            self.assertEqual(entries[0]["pair_path"], pair_path.resolve())
            self.assertEqual(summary.missing_pair, 0)

    def test_dataset_loads_paired_sample_and_collate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_path = root / "mesh.obj"
            pair_path = root / "mesh_pair.pt"
            mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            torch.save(
                {
                    "bbox_xyz": torch.tensor([1.0, 1.5, 1.2], dtype=torch.float32),
                    "text_input_ids": torch.arange(77, dtype=torch.long).unsqueeze(0),
                    "text_attention_mask": torch.ones(1, 77, dtype=torch.long),
                    "shape_ids": torch.arange(1024, dtype=torch.long).unsqueeze(0),
                },
                pair_path,
            )

            dataset = ObjaverseDataset(
                root=str(root),
                entries=[
                    {
                        "mesh_path": mesh_path.resolve(),
                        "text": "paired prompt",
                        "pair_path": pair_path.resolve(),
                    }
                ],
            )

            sample = dataset[0]
            assert sample is not None
            self.assertEqual(sample["prompt_text"], "paired prompt")
            self.assertEqual(tuple(sample["bbox_xyz"].shape), (3,))
            self.assertEqual(tuple(sample["text_input_ids"].shape), (77,))
            self.assertEqual(tuple(sample["text_attention_mask"].shape), (77,))
            self.assertEqual(tuple(sample["shape_ids"].shape), (1024,))

            batch = collate_objaverse_batch([sample])

            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch["prompt_text"], ["paired prompt"])
            self.assertEqual(tuple(batch["bbox_xyz"].shape), (1, 3))
            self.assertEqual(tuple(batch["text_input_ids"].shape), (1, 77))
            self.assertEqual(tuple(batch["text_attention_mask"].shape), (1, 77))
            self.assertEqual(tuple(batch["shape_ids"].shape), (1, 1024))
            self.assertNotIn("point_cloud", batch)

    def test_normalize_bbox_scales_by_max_axis(self) -> None:
        bbox = normalize_bbox((1.0, 2.0, 0.5))
        self.assertAlmostEqual(max(bbox), 1.925)
        self.assertAlmostEqual(bbox[0], 1.925 * 0.5)

    def test_prepare_sample_eval_specs_extracts_prompt_and_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_path = root / "tri.obj"
            mesh_path.write_text(
                "\n".join(
                    [
                        "v 0 0 0",
                        "v 2 0 0",
                        "v 0 1 0",
                        "v 0 0 0.5",
                        "f 1 2 3",
                        "f 1 2 4",
                    ]
                )
                + "\n"
            )

            specs = prepare_sample_eval_specs(
                [{"mesh_path": mesh_path, "text": "test prompt"}],
                max_samples=1,
            )

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].prompt_text, "test prompt")
            self.assertEqual(specs[0].mesh_path, str(mesh_path.resolve()))
            self.assertEqual(len(specs[0].bbox_xyz), 3)
            self.assertAlmostEqual(max(specs[0].bbox_xyz), 1.925)


if __name__ == "__main__":
    unittest.main()
