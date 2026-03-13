import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cube3d.train_block_diffusion import (
    _build_parser,
    _load_sample_eval_specs,
    _load_training_config_defaults,
    _save_sample_eval_specs,
    _seed_worker,
    _training_budget_summary,
    _validate_required_training_args,
    _write_entries_manifest,
    parse_args,
)
from cube3d.training import SampleEvalSpec
from cube3d.training.block_diffusion import validate_training_state_compatibility


class TrainEntrypointHelpersTest(unittest.TestCase):
    def test_seed_worker_is_deterministic(self) -> None:
        torch.manual_seed(1234)
        _seed_worker(0)
        sample_a = (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(()).item()),
        )

        torch.manual_seed(1234)
        _seed_worker(0)
        sample_b = (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(()).item()),
        )

        self.assertEqual(sample_a, sample_b)

    def test_write_entries_manifest_serializes_resolved_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_path = root / "mesh.obj"
            pair_path = root / "mesh_pair.pt"
            mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            pair_path.write_bytes(b"placeholder")

            manifest_path = _write_entries_manifest(
                root / "entries.jsonl",
                [{"mesh_path": mesh_path, "text": "prompt", "pair_path": pair_path}],
            )

            records = [
                json.loads(line)
                for line in manifest_path.read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["mesh_path"], str(mesh_path.resolve()))
            self.assertEqual(records[0]["pair_path"], str(pair_path.resolve()))
            self.assertEqual(records[0]["text"], "prompt")

    def test_sample_eval_specs_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [
                SampleEvalSpec(
                    prompt_text="prompt",
                    bbox_xyz=[1.0, 1.5, 1.2],
                    mesh_path="/tmp/example.obj",
                )
            ]

            path = _save_sample_eval_specs(Path(tmpdir) / "sample_eval_specs.json", specs)
            loaded = _load_sample_eval_specs(path)

            self.assertEqual(loaded, specs)

    def test_load_training_config_defaults_reads_known_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "batch_size: 2",
                        "save_full_state_interval: 123",
                        "output_dir: /tmp/run",
                    ]
                )
                + "\n"
            )

            parser = _build_parser()
            parser.add_argument("--batch-size", dest="batch_size", type=int, default=1)
            parser.add_argument(
                "--save-full-state-interval",
                dest="save_full_state_interval",
                type=int,
                default=0,
            )
            parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
            defaults = _load_training_config_defaults(parser, str(config_path))

            self.assertEqual(defaults["batch_size"], 2)
            self.assertEqual(defaults["save_full_state_interval"], 123)
            self.assertEqual(defaults["output_dir"], "/tmp/run")

    def test_validate_required_training_args_requires_budget(self) -> None:
        class Args:
            data_root = "/tmp/data"
            output_dir = "/tmp/out"
            max_steps = None
            epochs = None

        with self.assertRaisesRegex(ValueError, "training budget"):
            _validate_required_training_args(Args())

    def test_training_budget_summary_prefers_steps(self) -> None:
        class Args:
            max_steps = 1000
            epochs = 3

        summary = _training_budget_summary(Args())
        self.assertEqual(summary["budget_mode"], "steps")
        self.assertEqual(summary["max_steps"], 1000)

    def test_parse_args_reads_ddp_and_tensorboard_defaults_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "ddp_backend: gloo",
                        "tensorboard_dir: /tmp/tensorboard",
                        "disable_tensorboard: true",
                        "tensorboard_flush_secs: 99",
                        "model_dtype: bfloat16",
                        "activation_checkpointing: true",
                        "offload_shape_model_to_cpu: true",
                    ]
                )
                + "\n"
            )

            with patch(
                "sys.argv",
                [
                    "train_block_diffusion.py",
                    "--train-config-path",
                    str(config_path),
                ],
            ):
                args = parse_args()

            self.assertEqual(args.ddp_backend, "gloo")
            self.assertEqual(args.tensorboard_dir, "/tmp/tensorboard")
            self.assertTrue(args.disable_tensorboard)
            self.assertEqual(args.tensorboard_flush_secs, 99)
            self.assertEqual(args.model_dtype, "bfloat16")
            self.assertTrue(args.activation_checkpointing)
            self.assertTrue(args.offload_shape_model_to_cpu)


class TrainingStateCompatibilityTest(unittest.TestCase):
    def _expected_signature(self) -> dict:
        return {
            "block_size": 8,
            "mask_token_id": -1,
            "train_t_min": 0.45,
            "train_t_max": 0.95,
            "grad_accum_steps": 2,
            "warmup_steps": 10,
            "min_lr_ratio": 0.1,
            "amp_dtype": "bfloat16",
            "model_dtype": "float32",
            "shape_ckpt_path": "/tmp/shape.safetensors",
            "gpt_ckpt_path": "/tmp/gpt_latest.safetensors",
        }

    def _loaded_state(self) -> dict:
        return {
            "block_size": 8,
            "mask_token_id": -1,
            "train_t_min": 0.45,
            "train_t_max": 0.95,
            "grad_accum_steps": 2,
            "warmup_steps": 10,
            "min_lr_ratio": 0.1,
            "amp_dtype": "bfloat16",
            "model_dtype": "float32",
            "shape_ckpt_path": "/tmp/shape.safetensors",
            "gpt_ckpt_path": "/tmp/gpt_initial.safetensors",
            "model_checkpoint_path": "/tmp/gpt_latest.safetensors",
        }

    def test_validate_training_state_compatibility_accepts_matching_state(self) -> None:
        validate_training_state_compatibility(
            expected=self._expected_signature(),
            loaded=self._loaded_state(),
        )

    def test_validate_training_state_compatibility_rejects_mismatch(self) -> None:
        loaded = self._loaded_state()
        loaded["grad_accum_steps"] = 4

        with self.assertRaisesRegex(ValueError, "grad_accum_steps"):
            validate_training_state_compatibility(
                expected=self._expected_signature(),
                loaded=loaded,
            )


if __name__ == "__main__":
    unittest.main()
