import unittest

import torch

from cube3d.model.gpt.dual_stream_roformer import DualStreamRoformer


class DualStreamRoformerActivationCheckpointingTest(unittest.TestCase):
    def _build_model(self, activation_checkpointing: bool) -> DualStreamRoformer:
        model = DualStreamRoformer(
            DualStreamRoformer.Config(
                n_layer=2,
                n_single_layer=1,
                n_head=2,
                n_embd=16,
                bias=True,
                eps=1e-6,
                shape_model_vocab_size=32,
                shape_model_embed_dim=8,
                text_model_embed_dim=8,
                use_pooled_text_embed=False,
                encoder_with_cls_token=True,
                use_bbox=False,
                generation_mode="block_diffusion",
                block_size=2,
                activation_checkpointing=activation_checkpointing,
            )
        )
        model.train()
        return model

    def test_activation_checkpointing_preserves_outputs_and_gradients(self) -> None:
        torch.manual_seed(1234)
        baseline = self._build_model(activation_checkpointing=False)
        checkpointed = self._build_model(activation_checkpointing=True)
        checkpointed.load_state_dict(baseline.state_dict())

        embed = torch.randn(2, 4, 16)
        cond = torch.randn(2, 3, 16)
        attn_mask = torch.ones(7, 7, dtype=torch.bool)

        baseline_embed = embed.clone().requires_grad_(True)
        baseline_cond = cond.clone().requires_grad_(True)
        checkpoint_embed = embed.clone().requires_grad_(True)
        checkpoint_cond = cond.clone().requires_grad_(True)

        baseline_logits = baseline(
            embed=baseline_embed,
            cond=baseline_cond,
            attn_mask=attn_mask,
            use_single_blocks=True,
        )
        checkpoint_logits = checkpointed(
            embed=checkpoint_embed,
            cond=checkpoint_cond,
            attn_mask=attn_mask,
            use_single_blocks=True,
        )

        self.assertTrue(torch.allclose(baseline_logits, checkpoint_logits, atol=1e-5))

        baseline_loss = baseline_logits.square().mean()
        checkpoint_loss = checkpoint_logits.square().mean()
        baseline_loss.backward()
        checkpoint_loss.backward()

        self.assertTrue(
            torch.allclose(baseline_embed.grad, checkpoint_embed.grad, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(baseline_cond.grad, checkpoint_cond.grad, atol=1e-5)
        )

        for baseline_param, checkpoint_param in zip(
            baseline.parameters(),
            checkpointed.parameters(),
        ):
            if baseline_param.grad is None and checkpoint_param.grad is None:
                continue
            self.assertIsNotNone(baseline_param.grad)
            self.assertIsNotNone(checkpoint_param.grad)
            self.assertTrue(
                torch.allclose(
                    baseline_param.grad,
                    checkpoint_param.grad,
                    atol=1e-5,
                )
            )


if __name__ == "__main__":
    unittest.main()
