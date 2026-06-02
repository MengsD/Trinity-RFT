"""Test that prompt truncation produces dummy experiences with valid routed_experts.

Verifies that _handle_prompt_truncation fills routed_experts for dummy experiences
so they can be mixed with normal experiences without causing ValueError in to_data_proto.
"""

import asyncio
import unittest
from unittest import mock

import torch

from trinity.common.config import InferenceModelConfig
from trinity.common.experience import EID, Experience
from trinity.common.models.sglang_model import SGLangRolloutModel
from trinity.trainer.verl.utils import to_data_proto

NUM_LAYERS = 40
TOPK = 8
MAX_PROMPT_TOKENS = 10


def _create_model():
    """Create an SGLangRolloutModel instance with mocked tokenizer for testing."""
    config = InferenceModelConfig(
        model_path="mock_model",
        engine_type="sglang",
        enable_prompt_truncation=True,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        enable_return_routed_experts=True,
    )

    model = SGLangRolloutModel.__new__(SGLangRolloutModel)
    model.config = config
    model.logger = mock.Mock()
    model.chat_template = None
    model.enable_thinking = False
    model.api_client = mock.Mock()
    model._prepared = True

    mock_tokenizer = mock.Mock()
    long_prompt_ids = list(range(MAX_PROMPT_TOKENS + 5))  # 15 tokens > 10 max
    mock_tokenizer.return_value = {"input_ids": [torch.tensor(long_prompt_ids)]}
    mock_tokenizer.decode = mock.Mock(return_value="truncated prompt")
    model.tokenizer = mock_tokenizer

    return model


def make_normal_experience(seq_len=100, prompt_len=20):
    """Create a normal experience with valid routed_experts."""
    tokens = torch.randint(0, 1000, (seq_len,), dtype=torch.int32)
    routed_experts = torch.randint(0, 64, (seq_len - 1, NUM_LAYERS, TOPK), dtype=torch.uint8)
    logprobs = torch.randn(seq_len - prompt_len, dtype=torch.float32)
    return Experience(
        eid=EID(batch=1, task=1, run=1, step=1),
        tokens=tokens,
        logprobs=logprobs,
        prompt_length=prompt_len,
        routed_experts=routed_experts,
        reward=1.0,
    )


class TestRoutedExpertWithDummyExperience(unittest.TestCase):
    """Test routed_experts handling for dummy (truncated) experiences."""

    @mock.patch(
        "trinity.common.models.model.get_routed_experts_layout", return_value=(NUM_LAYERS, TOPK)
    )
    def test_handle_prompt_truncation_fills_routed_experts(self, mock_layout):
        """Verify _handle_prompt_truncation produces experiences with valid routed_experts."""
        model = _create_model()

        experiences, is_valid = model._handle_prompt_truncation("long prompt", n=8)

        self.assertFalse(is_valid)
        self.assertEqual(len(experiences), 8)
        for exp in experiences:
            self.assertEqual(exp.truncate_status, "prompt_truncated")
            self.assertIsNotNone(exp.routed_experts)
            self.assertEqual(exp.routed_experts.dtype, torch.uint8)
            expected_shape = (MAX_PROMPT_TOKENS, NUM_LAYERS, TOPK)
            self.assertEqual(tuple(exp.routed_experts.shape), expected_shape)
            self.assertEqual(exp.routed_experts.sum().item(), 0)

    @mock.patch(
        "trinity.common.models.model.get_routed_experts_layout", return_value=(NUM_LAYERS, TOPK)
    )
    def test_mixed_batch_succeeds(self, mock_layout):
        """Verify to_data_proto succeeds when mixing normal + truncated experiences."""
        model = _create_model()

        truncated_exps, is_valid = model._handle_prompt_truncation("long prompt", n=8)

        self.assertFalse(is_valid)
        for exp in truncated_exps:
            self.assertEqual(exp.truncate_status, "prompt_truncated")

        normal_exps = [make_normal_experience() for _ in range(4)]
        batch = normal_exps + truncated_exps

        result = to_data_proto(batch, pad_token_id=0, model=object(), logger=mock.Mock())

        self.assertIn("routed_experts", result.batch)
        routed_experts = result.batch["routed_experts"]
        self.assertEqual(routed_experts.dtype, torch.uint8)
        self.assertEqual(routed_experts.shape[0], 12)
        self.assertEqual(routed_experts.shape[2], NUM_LAYERS)
        self.assertEqual(routed_experts.shape[3], TOPK)

    @mock.patch(
        "trinity.common.models.model.get_routed_experts_layout", return_value=(NUM_LAYERS, TOPK)
    )
    def test_end_to_end_through_generate(self, mock_layout):
        model = _create_model()

        truncated_exps = list(asyncio.run(model.generate("long prompt", n=8)))

        for exp in truncated_exps:
            self.assertEqual(exp.truncate_status, "prompt_truncated")

        normal_exps = [make_normal_experience() for _ in range(4)]
        batch = normal_exps + truncated_exps

        result = to_data_proto(batch, pad_token_id=0, model=object(), logger=mock.Mock())
        self.assertIn("routed_experts", result.batch)
        self.assertEqual(result.batch["routed_experts"].shape[0], 12)


if __name__ == "__main__":
    unittest.main()
