import unittest

import torch

from cube3d.model.gpt.block_diffusion_utils import (
    build_training_shape_attention_mask,
    wrap_shape_attention_with_condition_prefix,
)


class BlockDiffusionMaskTest(unittest.TestCase):
    def test_training_mask_respects_clean_and_noisy_visibility(self) -> None:
        mask = build_training_shape_attention_mask(
            num_shape_tokens=8,
            block_size=4,
            device=torch.device("cpu"),
        )

        clean0 = slice(0, 4)
        clean1 = slice(4, 8)
        noisy0 = slice(8, 12)
        noisy1 = slice(12, 16)

        self.assertTrue(mask[clean0, clean0].all())
        self.assertFalse(mask[clean0, clean1].any())
        self.assertFalse(mask[clean0, noisy0].any())
        self.assertFalse(mask[clean0, noisy1].any())

        self.assertTrue(mask[clean1, clean0].all())
        self.assertTrue(mask[clean1, clean1].all())
        self.assertFalse(mask[clean1, noisy0].any())
        self.assertFalse(mask[clean1, noisy1].any())

        self.assertFalse(mask[noisy0, clean0].any())
        self.assertFalse(mask[noisy0, clean1].any())
        self.assertTrue(mask[noisy0, noisy0].all())
        self.assertFalse(mask[noisy0, noisy1].any())

        self.assertTrue(mask[noisy1, clean0].all())
        self.assertFalse(mask[noisy1, clean1].any())
        self.assertFalse(mask[noisy1, noisy0].any())
        self.assertTrue(mask[noisy1, noisy1].all())

    def test_condition_prefix_only_exposes_condition_to_shape_rows(self) -> None:
        shape_mask = torch.tensor(
            [
                [True, False],
                [True, True],
            ],
            dtype=torch.bool,
        )
        full_mask = wrap_shape_attention_with_condition_prefix(shape_mask, cond_len=2)

        self.assertEqual(tuple(full_mask.shape), (4, 4))
        self.assertTrue(full_mask[:2, :2].all())
        self.assertFalse(full_mask[:2, 2:].any())
        self.assertTrue(full_mask[2:, :2].all())
        self.assertTrue(torch.equal(full_mask[2:, 2:], shape_mask))

    def test_full_training_mask_matches_cond_clean_noisy_contract(self) -> None:
        shape_mask = build_training_shape_attention_mask(
            num_shape_tokens=8,
            block_size=4,
            device=torch.device("cpu"),
        )
        full_mask = wrap_shape_attention_with_condition_prefix(shape_mask, cond_len=2)

        cond = slice(0, 2)
        clean0 = slice(2, 6)
        clean1 = slice(6, 10)
        noisy0 = slice(10, 14)
        noisy1 = slice(14, 18)

        self.assertTrue(full_mask[cond, cond].all())
        self.assertFalse(full_mask[cond, clean0].any())
        self.assertFalse(full_mask[cond, noisy0].any())

        self.assertTrue(full_mask[clean0, cond].all())
        self.assertTrue(full_mask[clean0, clean0].all())
        self.assertFalse(full_mask[clean0, clean1].any())
        self.assertFalse(full_mask[clean0, noisy0].any())

        self.assertTrue(full_mask[clean1, cond].all())
        self.assertTrue(full_mask[clean1, clean0].all())
        self.assertTrue(full_mask[clean1, clean1].all())
        self.assertFalse(full_mask[clean1, noisy1].any())

        self.assertTrue(full_mask[noisy0, cond].all())
        self.assertFalse(full_mask[noisy0, clean0].any())
        self.assertFalse(full_mask[noisy0, clean1].any())
        self.assertTrue(full_mask[noisy0, noisy0].all())
        self.assertFalse(full_mask[noisy0, noisy1].any())

        self.assertTrue(full_mask[noisy1, cond].all())
        self.assertTrue(full_mask[noisy1, clean0].all())
        self.assertFalse(full_mask[noisy1, clean1].any())
        self.assertFalse(full_mask[noisy1, noisy0].any())
        self.assertTrue(full_mask[noisy1, noisy1].all())


if __name__ == "__main__":
    unittest.main()
