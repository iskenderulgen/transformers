import math
import tempfile
import unittest

import pytest

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertModel,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch
    from torch.optim import SGD


def _create_base_config(**overrides):
    config = BertConfig(
        vocab_size=101,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=None,
        hidden_act="silu",
        use_alibi=True,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@require_torch
class BertModelForwardTest(unittest.TestCase):
    def test_forward_shape_and_mask(self):
        config = _create_base_config()
        model = BertModel(config).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 12), device=torch_device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertEqual(outputs.last_hidden_state.shape, (2, 12, config.hidden_size))
        self.assertIsNone(outputs.attentions)

    def test_context_extrapolation(self):
        config = _create_base_config()
        model = BertModel(config).to(torch_device)
        long_input = torch.randint(0, config.vocab_size, (1, 192), device=torch_device)
        outputs = model(long_input)
        self.assertEqual(outputs.last_hidden_state.shape, (1, 192, config.hidden_size))

    def test_structure_matches_expectations(self):
        config = _create_base_config()
        model = BertModel(config).to(torch_device)
        self.assertFalse(hasattr(model.embeddings, "position_embeddings"))
        for layer in model.encoder.layer:
            self.assertTrue(hasattr(layer.pre_attention_layernorm, "weight"))
            self.assertTrue(hasattr(layer.intermediate, "gate_proj"))
            self.assertTrue(hasattr(layer.intermediate, "up_proj"))
        slopes = model.encoder.layer[0].attention.self.alibi_slopes
        self.assertTrue(torch.all(slopes > 0))
        self.assertTrue(torch.all(slopes[:-1] >= slopes[1:]))

    def test_save_and_load_round_trip(self):
        config = _create_base_config()
        model = BertModel(config).to(torch_device).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=torch_device)
        outputs_before = model(input_ids).last_hidden_state.detach()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            reloaded = BertModel.from_pretrained(tmpdir).to(torch_device).eval()

        outputs_after = reloaded(input_ids).last_hidden_state.detach()
        self.assertTrue(torch.allclose(outputs_before, outputs_after, atol=1e-6))
    def test_forward_dtypes(self):
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        for dtype in dtypes:
            if dtype in {torch.float16, torch.bfloat16} and torch_device == "cpu":
                continue
            config = _create_base_config()
            model = BertModel(config).to(torch_device).to(dtype=dtype)
            model.eval()
            input_ids = torch.randint(0, config.vocab_size, (2, 32), device=torch_device)
            device_type = torch_device.split(":")[0]
            use_autocast = dtype != torch.float32
            autocast_context = torch.autocast(device_type=device_type, dtype=dtype, enabled=use_autocast)
            with autocast_context:
                outputs = model(input_ids)
            self.assertFalse(torch.isnan(outputs.last_hidden_state).any())
            self.assertFalse(torch.isinf(outputs.last_hidden_state).any())

    @pytest.mark.skipif(torch.__version__ < "2.1", reason="torch.compile requires torch 2.1+")
    def test_torch_compile_forward_backward(self):
        if torch_device == "cpu":
            backend = "inductor"
        else:
            backend = "inductor"
        config = _create_base_config()
        model = BertForMaskedLM(config).to(torch_device)
        compiled_model = torch.compile(model, backend=backend)
        compiled_model.train()
        optimizer = SGD(compiled_model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=torch_device)
        labels = input_ids.clone()
        optimizer.zero_grad()
        loss = compiled_model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()

    def test_dropout_determinism(self):
        config = _create_base_config()
        model = BertModel(config).to(torch_device)
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=torch_device)
        model.train()
        torch.manual_seed(0)
        out1 = model(input_ids).last_hidden_state
        torch.manual_seed(0)
        out2 = model(input_ids).last_hidden_state
        self.assertTrue(torch.allclose(out1, out2))
        model.eval()
        out_eval1 = model(input_ids).last_hidden_state
        out_eval2 = model(input_ids).last_hidden_state
        self.assertTrue(torch.allclose(out_eval1, out_eval2))


@require_torch
class BertTrainingStepTest(unittest.TestCase):
    def test_mlm_backward_and_step(self):
        config = _create_base_config()
        model = BertForMaskedLM(config).to(torch_device)
        model.train()
        optimizer = SGD(model.parameters(), lr=1e-3)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=torch_device)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=torch_device)
        mask = torch.rand(batch_size, seq_len, device=torch_device) < 0.2
        labels = labels.masked_fill(~mask, -100)
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        self.assertTrue(all(param.grad is not None for param in model.parameters() if param.requires_grad))
        optimizer.step()

    def test_sequence_classification_backward(self):
        config = _create_base_config(num_labels=3)
        model = BertForSequenceClassification(config).to(torch_device)
        model.train()
        optimizer = SGD(model.parameters(), lr=1e-3)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=torch_device)
        labels = torch.randint(0, config.num_labels, (batch_size,), device=torch_device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()
        optimizer.step()
    def test_gradient_checkpointing_step(self):
        config = _create_base_config()
        model = BertForMaskedLM(config).to(torch_device)
        model.gradient_checkpointing_enable()
        model.train()
        optimizer = SGD(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, config.vocab_size, (2, 12), device=torch_device)
        labels = input_ids.clone()
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()

    def test_autocast_long_context(self):
        if torch_device == "cpu":
            self.skipTest("autocast long context test requires CUDA")
        config = _create_base_config()
        model = BertModel(config).to(torch_device)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 1024), device=torch_device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids)
        self.assertEqual(outputs.last_hidden_state.shape, (1, 1024, config.hidden_size))


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch is required")
class BertConfigValidationTest(unittest.TestCase):
    def test_default_intermediate_defaults(self):
        config = BertConfig(hidden_size=96)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.intermediate_size, int(math.ceil((8 * config.hidden_size) / 3)))
        self.assertTrue(config.use_alibi)

    def test_head_dimension_validation(self):
        config = BertConfig(
            hidden_size=48,
            num_attention_heads=8,
            use_alibi=True,
        )
        with self.assertRaises(ValueError):
            BertModel(config)
