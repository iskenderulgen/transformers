# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import pytest
from packaging import version

from transformers import AutoTokenizer, BertConfig, is_torch_available
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.auto import get_values
from transformers.testing_utils import (
    CaptureLogger,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        BertForMaskedLM,
        BertForMultipleChoice,
        BertForNextSentencePrediction,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
        BertLMHeadModel,
        BertModel,
        logging,
    )


class BertModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=None,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        """
        Returns a tiny configuration by default.
        """
        return BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )


    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))


    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))


    def create_and_check_for_next_sequence_prediction(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BertForNextSentencePrediction(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, 2))

    def create_and_check_for_pretraining(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BertForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
            next_sentence_label=sequence_labels,
        )
        self.parent.assertEqual(result.prediction_logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertEqual(result.seq_relationship_logits.shape, (self.batch_size, 2))

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BertForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = BertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = BertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = BertForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class BertFlexAttentionTest(unittest.TestCase):
    def _make_config(self, seq_length: int = 6, **kwargs):
        defaults = {
            "vocab_size": 32,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 32,
            "attention_probs_dropout_prob": 0.0,
            "hidden_dropout_prob": 0.0,
        }
        defaults.update(kwargs)
        return BertConfig(**defaults)

    # ===== Basic Functionality Tests =====

    def test_forward_with_document_ids(self):
        """Test that model handles document_ids for packed sequences correctly."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        input_ids = ids_tensor([1, 6], config.vocab_size).to(torch_device)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]], device=torch_device)
        document_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], device=torch_device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids)
        self.assertEqual(outputs.last_hidden_state.shape, (1, 6, config.hidden_size))
        self.assertIsNotNone(outputs.pooler_output)

    def test_backward_runs(self):
        """Test that gradients flow correctly through the model."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        input_ids = ids_tensor([2, 5], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids, device=torch_device)
        document_ids = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 2, 2]], device=torch_device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids)
        loss = outputs.last_hidden_state.sum()
        loss.backward()

        # Verify gradients populated throughout the model
        self.assertIsNotNone(model.embeddings.word_embeddings.weight.grad)
        self.assertIsNotNone(model.encoder.layer[0].attention.self.query.weight.grad)
        self.assertIsNotNone(model.encoder.layer[0].intermediate.gate_value_proj.weight.grad)

    def test_block_mask_changes_with_documents(self):
        """Test that different document boundaries produce different outputs."""
        config = self._make_config(seq_length=4)
        model = BertModel(config).to(torch_device)
        input_ids = torch.arange(4, device=torch_device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, device=torch_device)
        doc_a = torch.tensor([[0, 0, 1, 1]], device=torch_device)
        doc_b = torch.tensor([[0, 1, 0, 1]], device=torch_device)

        out_a = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=doc_a).last_hidden_state
        out_b = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=doc_b).last_hidden_state
        self.assertFalse(torch.allclose(out_a, out_b))

    # ===== Pre-RMSNorm Tests =====

    def test_rmsnorm_applied_before_attention(self):
        """Verify Pre-RMSNorm is applied before attention."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)

        # Check that RMSNorm layers exist
        for layer in model.encoder.layer:
            self.assertIsInstance(layer.attention_rmsnorm, torch.nn.RMSNorm)
            self.assertIsInstance(layer.output_rmsnorm, torch.nn.RMSNorm)

        # Check embeddings RMSNorm exists
        self.assertIsInstance(model.embeddings_rmsnorm, torch.nn.RMSNorm)

    def test_rmsnorm_eps_value(self):
        """Test that RMSNorm uses correct epsilon from config."""
        config = self._make_config()
        config.layer_norm_eps = 1e-6
        model = BertModel(config).to(torch_device)

        # Check epsilon matches config
        self.assertEqual(model.embeddings_rmsnorm.eps, config.layer_norm_eps)
        self.assertEqual(model.encoder.layer[0].attention_rmsnorm.eps, config.layer_norm_eps)
        self.assertEqual(model.encoder.layer[0].output_rmsnorm.eps, config.layer_norm_eps)

    # ===== SwiGLU Tests =====

    def test_swiglu_has_packed_projection(self):
        """Verify SwiGLU has packed gate and value projection."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)

        intermediate = model.encoder.layer[0].intermediate
        self.assertTrue(hasattr(intermediate, 'gate_value_proj'))

        # Check projects to 2 * intermediate_size
        self.assertEqual(intermediate.gate_value_proj.out_features, 2 * config.intermediate_size)

    def test_swiglu_output_shape(self):
        """Test SwiGLU intermediate layer output shape."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=torch_device)

        intermediate = model.encoder.layer[0].intermediate
        output = intermediate(hidden_states)

        # Output should be intermediate_size
        self.assertEqual(output.shape, (batch_size, seq_len, config.intermediate_size))

    def test_intermediate_size_calculation(self):
        """Test that intermediate_size is computed correctly for SwiGLU."""
        config = BertConfig(hidden_size=768)
        expected_intermediate = int((8.0 / 3.0) * 768)  # 2048
        self.assertEqual(config.intermediate_size, expected_intermediate)
        self.assertEqual(config.ffn_activation, "swiglu")

        # Test custom intermediate_size
        config_custom = BertConfig(hidden_size=768, intermediate_size=3072)
        self.assertEqual(config_custom.intermediate_size, 3072)

    def test_invalid_ffn_activation_raises(self):
        """Test that unsupported FFN activations are rejected."""
        with self.assertRaises(ValueError):
            BertConfig(ffn_activation="gelu")

    # ===== Gradient Checkpointing Tests =====

    def test_gradient_checkpointing_enable(self):
        """Test enabling gradient checkpointing."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        self.assertTrue(model.encoder.gradient_checkpointing)

    def test_gradient_checkpointing_training(self):
        """Test gradient checkpointing during training."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.gradient_checkpointing_enable()
        model.train()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids, device=torch_device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.last_hidden_state.sum()
        loss.backward()

        # Verify gradients still flow correctly with checkpointing
        self.assertIsNotNone(model.embeddings.word_embeddings.weight.grad)

    def test_gradient_checkpointing_with_document_ids(self):
        """Ensure gradient checkpointing still works when using document masking."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.gradient_checkpointing_enable()
        model.train()

        input_ids = ids_tensor([2, 5], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids, device=torch_device)
        document_ids = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 2, 2]], device=torch_device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids)
        loss = outputs.last_hidden_state.sum()
        loss.backward()

        self.assertIsNotNone(model.embeddings.word_embeddings.weight.grad)

    # ===== Document Masking Tests =====

    def test_document_masking_prevents_cross_attention(self):
        """Test that tokens from different documents don't attend to each other."""
        config = self._make_config(num_hidden_layers=1, hidden_size=32, num_attention_heads=2)
        model = BertModel(config).to(torch_device)
        model.eval()

        # Create input with two documents
        input_ids = torch.tensor([[1, 2, 3, 4]], device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        document_ids = torch.tensor([[1, 1, 2, 2]], device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids)

        # Output should be different from no document masking
        with torch.no_grad():
            outputs_no_doc = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertFalse(torch.allclose(outputs.last_hidden_state, outputs_no_doc.last_hidden_state))

    def test_document_masking_with_padding(self):
        """Test document masking works correctly with padding."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 0, 0, 0]], device=torch_device)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]], device=torch_device)
        document_ids = torch.tensor([[0, 0, 1, -1, -1, -1]], device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids)

        self.assertEqual(outputs.last_hidden_state.shape, (1, 6, config.hidden_size))

    def test_packed_sequences_batch(self):
        """Test packed sequences with multiple documents per batch."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        batch_size, seq_len = 3, 8
        input_ids = ids_tensor([batch_size, seq_len], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        # Different packing patterns per batch
        document_ids = torch.tensor([
            [1, 1, 1, 2, 2, 2, 3, 3],  # 3 documents
            [1, 1, 1, 1, 2, 2, 2, 2],  # 2 documents
            [1, 1, 1, 1, 1, 1, 1, 1],  # 1 document
        ], device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids)

        self.assertEqual(outputs.last_hidden_state.shape, (batch_size, seq_len, config.hidden_size))

    def test_single_document_sequence(self):
        """Test packed sequence with only one document (no masking needed)."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)
        # All tokens belong to the same document
        document_ids = torch.zeros_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids)

        self.assertEqual(outputs.last_hidden_state.shape, (2, 8, config.hidden_size))

    def test_all_different_documents(self):
        """Test extreme case where each token is a different document."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        batch_size, seq_len = 2, 6
        input_ids = ids_tensor([batch_size, seq_len], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)
        # Each token is its own document
        document_ids = torch.arange(seq_len, device=torch_device).unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids)

        self.assertEqual(outputs.last_hidden_state.shape, (batch_size, seq_len, config.hidden_size))

    def test_document_ids_with_mixed_padding(self):
        """Test document_ids with different padding patterns per batch."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        input_ids = torch.tensor([
            [1, 2, 3, 4, 0, 0],  # 4 tokens, 2 padding
            [1, 2, 3, 0, 0, 0],  # 3 tokens, 3 padding
        ], device=torch_device)
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ], device=torch_device)
        document_ids = torch.tensor([
            [0, 0, 1, 1, -1, -1],  # 2 docs, padding has doc_id -1
            [0, 1, 1, -1, -1, -1],  # 2 docs, padding has doc_id -1
        ], device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids)

        self.assertEqual(outputs.last_hidden_state.shape, (2, 6, config.hidden_size))

    def test_document_ids_none_defaults_to_single_document(self):
        """Test that document_ids=None is equivalent to all tokens in one document."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs_no_doc = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use document ID 0 for all tokens (padding would be -1 if present)
            outputs_single_doc = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                document_ids=torch.zeros_like(input_ids)
            )

        # Should produce identical results
        torch.testing.assert_close(outputs_no_doc.last_hidden_state, outputs_single_doc.last_hidden_state)

    def test_document_masking_independence(self):
        """
        Test that document masking prevents cross-document attention.
        Tokens in different documents should not attend to each other.
        """
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        # Create a packed sequence with two documents
        input_ids = torch.tensor([[1, 2, 3, 4]], device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        
        # Test 1: Two separate documents [Doc0, Doc0, Doc1, Doc1]
        document_ids_split = torch.tensor([[0, 0, 1, 1]], device=torch_device)
        
        # Test 2: All tokens in one document [Doc0, Doc0, Doc0, Doc0]
        document_ids_single = torch.tensor([[0, 0, 0, 0]], device=torch_device)

        with torch.no_grad():
            out_split = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids_split).last_hidden_state
            out_single = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids_single).last_hidden_state

        # Outputs should be different because attention patterns differ
        # With split docs: token[0] can't see token[2,3], token[2] can't see token[0,1]
        # With single doc: all tokens can see each other
        self.assertFalse(torch.allclose(out_split, out_single))

    def test_standard_bert_input_behavior(self):
        """
        Test that the model behaves correctly with standard [CLS] ... [SEP] input
        when no document_ids are provided (defaulting to single document).
        """
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        # [CLS] hello world [SEP] (token ids kept tiny to fit the synthetic vocab)
        cls_token, hello_token, world_token, sep_token = 1, 5, 6, 2
        input_ids = torch.tensor([[cls_token, hello_token, world_token, sep_token]], device=torch_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            # Case 1: No document_ids (standard usage)
            output_default = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            # Case 2: Explicit single document
            document_ids = torch.zeros_like(input_ids)
            output_explicit = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids).last_hidden_state

        # Should be exactly the same
        torch.testing.assert_close(output_default, output_explicit)

        # Also verify shape and non-NaN
        self.assertEqual(output_default.shape, (1, 4, config.hidden_size))
        self.assertFalse(torch.isnan(output_default).any())

    # ===== Unsupported Inputs Tests =====

    def test_head_mask_not_supported(self):
        """Verify that passing head_mask raises an explicit error."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        head_mask = torch.ones((config.num_hidden_layers, config.num_attention_heads), device=torch_device)

        input_ids = ids_tensor([1, 4], config.vocab_size).to(torch_device)

        with self.assertRaises(ValueError):
            model(input_ids=input_ids, head_mask=head_mask)

    def test_use_cache_not_supported(self):
        """Verify that use_cache triggers a clear error."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        input_ids = ids_tensor([1, 4], config.vocab_size).to(torch_device)

        with self.assertRaises(ValueError):
            model(input_ids=input_ids, use_cache=True)

    def test_cross_attention_not_supported(self):
        """Verify that encoder_hidden_states are rejected."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        input_ids = ids_tensor([1, 4], config.vocab_size).to(torch_device)
        encoder_hidden_states = torch.randn(1, 4, config.hidden_size, device=torch_device)

        with self.assertRaises(ValueError):
            model(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)

    # ===== Configuration Validation Tests =====

    def test_relative_position_embeddings_rejected(self):
        """Test that relative position embeddings are rejected."""
        with self.assertRaises(ValueError) as cm:
            BertConfig(position_embedding_type="relative_key")
        self.assertIn("only supports 'absolute'", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            BertConfig(position_embedding_type="relative_key_query")
        self.assertIn("only supports 'absolute'", str(cm.exception))

    def test_attention_dropout_warning(self):
        """Test that attention dropout > 0 triggers warning."""
        config = BertConfig(attention_probs_dropout_prob=0.1)
        # Creating model should log warning about dropout not being supported
        model = BertModel(config).to(torch_device)
        # Verify dropout is set to 0
        self.assertEqual(model.encoder.layer[0].attention.self.dropout_prob, 0.0)

    def test_decoder_mode_rejected_at_config(self):
        """Test that is_decoder=True is rejected during config creation."""
        with self.assertRaises(ValueError) as cm:
            BertConfig(is_decoder=True)
        self.assertIn("encoder-only", str(cm.exception).lower())

    def test_decoder_mode_rejected_at_model_init(self):
        """Test that is_decoder=True is rejected during model initialization."""
        config = self._make_config()
        # Manually set is_decoder after config creation to bypass config validation
        config.is_decoder = True
        with self.assertRaises(ValueError) as cm:
            BertModel(config).to(torch_device)
        self.assertIn("encoder-only", str(cm.exception).lower())

    def test_cross_attention_rejected_at_config(self):
        """Test that add_cross_attention=True is rejected during config creation."""
        with self.assertRaises(ValueError) as cm:
            BertConfig(add_cross_attention=True)
        self.assertIn("encoder-only", str(cm.exception).lower())

    def test_cross_attention_rejected_at_model_init(self):
        """Test that add_cross_attention=True is rejected during model initialization."""
        config = self._make_config()
        # Manually set add_cross_attention after config creation
        config.add_cross_attention = True
        with self.assertRaises(ValueError) as cm:
            BertModel(config).to(torch_device)
        self.assertIn("encoder-only", str(cm.exception).lower())

    # ===== Model Variants Tests =====

    def test_masked_lm_model(self):
        """Test BertForMaskedLM works correctly."""
        config = self._make_config()
        model = BertForMaskedLM(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        labels = ids_tensor([2, 8], config.vocab_size).to(torch_device)

        outputs = model(input_ids=input_ids, labels=labels)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.logits.shape, (2, 8, config.vocab_size))

    def test_sequence_classification_model(self):
        """Test BertForSequenceClassification works correctly."""
        config = self._make_config()
        config.num_labels = 3
        model = BertForSequenceClassification(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        labels = ids_tensor([2], config.num_labels).to(torch_device)

        outputs = model(input_ids=input_ids, labels=labels)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.logits.shape, (2, config.num_labels))

    def test_token_classification_model(self):
        """Test BertForTokenClassification works correctly."""
        config = self._make_config()
        config.num_labels = 5
        model = BertForTokenClassification(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        labels = ids_tensor([2, 8], config.num_labels).to(torch_device)

        outputs = model(input_ids=input_ids, labels=labels)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.logits.shape, (2, 8, config.num_labels))

    # ===== Parameter Count Tests =====

    def test_parameter_count_swiglu(self):
        """Verify SwiGLU parameters (packed)."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)

        layer = model.encoder.layer[0]

        # SwiGLU has packed projection: gate_value_proj
        # Note: Bias is removed in the modernized variant
        params = layer.intermediate.gate_value_proj.weight.numel()

        # Should project from hidden_size to 2 * intermediate_size (weights only)
        expected_params = config.hidden_size * (2 * config.intermediate_size)
        self.assertEqual(params, expected_params)

    # ===== Numerical Stability Tests =====

    def test_rmsnorm_numerical_stability(self):
        """Test RMSNorm numerical stability with extreme values."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        # Test with very large values
        large_input = torch.randn(1, 4, config.hidden_size, device=torch_device) * 1000
        rmsnorm = model.embeddings_rmsnorm
        output = rmsnorm(large_input)

        # Output should be normalized (no NaN or Inf)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_mixed_precision_compatibility(self):
        """Test model works with automatic mixed precision."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        config = self._make_config()
        model = BertModel(config).cuda()
        model.train()

        input_ids = ids_tensor([2, 8], config.vocab_size).cuda()

        # Test with autocast
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids)
            loss = outputs.last_hidden_state.sum()

        loss.backward()
        self.assertIsNotNone(model.embeddings.word_embeddings.weight.grad)

    def test_mixed_precision_rms_norm_configurations(self):
        """
        Tests various combinations of AMP and RMSNorm dtype to ensure:
        1. Happy Path: AMP=ON (BF16) + RMSNorm=BF16 -> Works (End-to-end BF16, fused kernels)
        2. Instability A: AMP=OFF + RMSNorm=BF16 -> Fails (Linear layer dtype mismatch)
        3. Suboptimal: AMP=ON (BF16) + RMSNorm=FP32 -> Works but loses fused kernel benefit
        """
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            return

        # Case 1: Happy Path (AMP ON, RMSNorm BF16)
        # Expected: Pass, output is BF16, embeddings stay FP32, fused kernels enabled
        config = self._make_config()
        config.rms_norm_dtype = "bfloat16"
        model = BertModel(config).to(torch_device)
        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        
        # Verify embeddings are still FP32 for stability
        self.assertEqual(model.embeddings.word_embeddings.weight.dtype, torch.float32)
        # Verify RMSNorm is BF16
        self.assertEqual(model.embeddings_rmsnorm.weight.dtype, torch.bfloat16)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids)
        
        self.assertIsNotNone(outputs.last_hidden_state)
        self.assertEqual(outputs.last_hidden_state.dtype, torch.bfloat16)

        # Case 2: Instability A (AMP OFF, RMSNorm BF16)
        # Expected: Fail. RMSNorm outputs BF16, next Linear (FP32) crashes.
        config = self._make_config()
        config.rms_norm_dtype = "bfloat16"
        model = BertModel(config).to(torch_device) # Linear layers are FP32
        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)

        with self.assertRaises(RuntimeError):
            model(input_ids=input_ids)

        # Case 3: Suboptimal (AMP ON, RMSNorm FP32)
        # Expected: Works, but autocast handles dtype conversions (loses fused kernel benefit)
        # This is valid but not optimal for performance.
        config = self._make_config()
        config.rms_norm_dtype = torch.float32
        model = BertModel(config).to(torch_device)
        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids)
        
        # Output will be BF16 because of autocast, but RMSNorm doesn't benefit from fused kernels
        self.assertIsNotNone(outputs.last_hidden_state)
        self.assertEqual(outputs.last_hidden_state.dtype, torch.bfloat16)

    def test_dtype_configuration_warnings(self):
        """Test that warnings are raised for suboptimal dtype configurations."""
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            return

        # Case 1: AMP ON + RMSNorm FP32 (suboptimal - should warn)
        config = self._make_config()
        config.rms_norm_dtype = torch.float32
        model = BertModel(config).to(torch_device)
        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)

        with self.assertLogs(logger="transformers.models.bert.modeling_bert", level="WARNING") as cm:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(input_ids=input_ids)
        
        self.assertTrue(
            any("Autocast is enabled but RMSNorm is configured to use FP32" in msg for msg in cm.output),
            "Expected warning about suboptimal AMP+FP32 RMSNorm configuration"
        )

        # Case 2: AMP OFF + RMSNorm BF16 (broken - should warn before crash)
        config = self._make_config()
        config.rms_norm_dtype = "bfloat16"
        model = BertModel(config).to(torch_device)
        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)

        with self.assertLogs(logger="transformers.models.bert.modeling_bert", level="WARNING") as cm:
            try:
                model(input_ids=input_ids)
            except RuntimeError:
                pass  # Expected to fail, we just want to check the warning

        self.assertTrue(
            any("RMSNorm is configured to use" in msg and "but autocast is not enabled" in msg for msg in cm.output),
            "Expected warning about dtype mismatch when AMP is OFF but RMSNorm is BF16"
        )

    # ===== Serialization Tests =====

    def test_save_and_load_model(self):
        """Test that FlexBERT model can be saved and loaded correctly."""
        import tempfile
        import os

        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        # Get original outputs
        with torch.no_grad():
            original_outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save_pretrained(save_path)

            # Load model
            loaded_model = BertModel.from_pretrained(save_path).to(torch_device)
            loaded_model.eval()

            # Get loaded outputs
            with torch.no_grad():
                loaded_outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)

        # Outputs should be identical
        torch.testing.assert_close(original_outputs.last_hidden_state, loaded_outputs.last_hidden_state)

    def test_config_preservation_after_save_load(self):
        """Test that FlexBERT config is preserved after save/load."""
        import tempfile
        import os

        config = BertConfig(
            vocab_size=100,
            hidden_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=256,
            ffn_activation="swiglu",
            position_embedding_type="absolute",
        )
        model = BertModel(config).to(torch_device)

        # Save and load config
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save_pretrained(save_path)
            loaded_config = BertConfig.from_pretrained(save_path)

        # Verify critical config parameters
        self.assertEqual(loaded_config.ffn_activation, "swiglu")
        self.assertEqual(loaded_config.position_embedding_type, "absolute")
        self.assertEqual(loaded_config.intermediate_size, 256)
        self.assertEqual(loaded_config.hidden_size, 128)
        self.assertEqual(loaded_config.num_hidden_layers, 3)

    def test_state_dict_contains_swiglu_parameters(self):
        """Test that state dict contains gate_value_proj for SwiGLU."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)

        state_dict = model.state_dict()

        # Check that packed projection exists for each layer
        for layer_idx in range(config.num_hidden_layers):
            key = f"encoder.layer.{layer_idx}.intermediate.gate_value_proj.weight"
            self.assertIn(key, state_dict)

    def test_state_dict_contains_rmsnorm_parameters(self):
        """Test that state dict contains RMSNorm parameters."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)

        state_dict = model.state_dict()

        # Check embeddings RMSNorm
        self.assertIn("embeddings_rmsnorm.weight", state_dict)

        # Check encoder layer RMSNorms
        for layer_idx in range(config.num_hidden_layers):
            attention_rmsnorm_key = f"encoder.layer.{layer_idx}.attention_rmsnorm.weight"
            output_rmsnorm_key = f"encoder.layer.{layer_idx}.output_rmsnorm.weight"

            self.assertIn(attention_rmsnorm_key, state_dict)
            self.assertIn(output_rmsnorm_key, state_dict)

    def setUp(self):
        self.model_tester = BertModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        # This modernized BERT variant only supports absolute position embeddings
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config_and_inputs[0].position_embedding_type = "absolute"
        self.model_tester.create_and_check_model(*config_and_inputs)

        # Test that relative position embeddings are rejected at config creation time
        for pos_type in ["relative_key", "relative_key_query"]:
            with self.assertRaises(ValueError) as cm:
                BertConfig(position_embedding_type=pos_type)
            self.assertIn("only supports 'absolute'", str(cm.exception))

    def test_model_3d_mask_shapes(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        # manipulate input_mask
        config_and_inputs = list(config_and_inputs)
        batch_size, seq_length = config_and_inputs[3].shape
        config_and_inputs[3] = random_attention_mask([batch_size, seq_length, seq_length])
        with self.assertRaises(ValueError):
            self.model_tester.create_and_check_model(*config_and_inputs)


    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)


    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_next_sequence_prediction(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_next_sequence_prediction(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_warning_if_padding_and_no_attention_mask(self):
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.model_tester.prepare_config_and_inputs()

        # Set pad tokens in the input_ids
        input_ids[0, 0] = config.pad_token_id

        # Check for warnings if the attention_mask is missing.
        logger = logging.get_logger("transformers.modeling_utils")
        # clear cache so we can test the warning is emitted.
        logger.warning_once.cache_clear()

        with CaptureLogger(logger) as cl:
            model = BertModel(config=config)
            model.to(torch_device)
            model.eval()
            model(input_ids, attention_mask=None, token_type_ids=token_type_ids)
        self.assertIn("We strongly recommend passing in an `attention_mask`", cl.out)

    @slow
    def test_model_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        model = BertModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
class BertModelIntegrationTest(unittest.TestCase):
    def _make_config(self, **kwargs):
        return BertConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
        )

    @slow
    def test_inference_no_head_absolute_embedding(self):
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = torch.Size((1, 11, 768))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor([[[0.4249, 0.1008, 0.7531], [0.3771, 0.1188, 0.7467], [0.4152, 0.1098, 0.7108]]])

        torch.testing.assert_close(output[:, 1:4, 1:4], expected_slice, rtol=1e-4, atol=1e-4)

    def test_packed_sequences_with_special_tokens(self):
        """
        Test a realistic packed sequence scenario with [CLS] and [SEP] tokens.
        Verifies that document masking correctly isolates sequences in packed format:
        [CLS] A [SEP] [CLS] B [SEP] where each sequence has independent attention.
        
        Note: Position IDs are continuous (NOT reset) as per FlexAttention design.
        Document isolation is achieved through document_ids, not position resets.
        """
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        # Create a packed sequence: [CLS] 10 11 [SEP] [CLS] 20 21 [SEP]
        cls_token, sep_token = 1, 2
        packed_input_ids = torch.tensor([[
            cls_token, 10, 11, sep_token,  # Doc 0
            cls_token, 20, 21, sep_token   # Doc 1
        ]], device=torch_device)

        # Document IDs: [0, 0, 0, 0, 1, 1, 1, 1]
        packed_doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], device=torch_device)
        packed_mask = torch.ones_like(packed_input_ids)

        with torch.no_grad():
            packed_out = model(
                input_ids=packed_input_ids,
                attention_mask=packed_mask,
                document_ids=packed_doc_ids
            ).last_hidden_state

        # Verify: output shape is correct
        self.assertEqual(packed_out.shape, (1, 8, config.hidden_size))
        
        # Verify: no NaN or inf values
        self.assertFalse(torch.isnan(packed_out).any())
        self.assertFalse(torch.isinf(packed_out).any())
        
        # Verify: packed execution with different document patterns produces different results
        # All in one document
        single_doc_ids = torch.zeros_like(packed_doc_ids)
        with torch.no_grad():
            single_out = model(
                input_ids=packed_input_ids,
                attention_mask=packed_mask,
                document_ids=single_doc_ids
            ).last_hidden_state
        
        # Outputs should differ because attention patterns are different
        self.assertFalse(torch.allclose(packed_out, single_out))

    def test_position_ids_reset_with_document_ids(self):
        """Test that position_ids are reset based on document_ids."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)

        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=torch_device)
        # Document IDs: [0, 0, 1, 1, 1] -> Doc 0 has length 2, Doc 1 has length 3
        document_ids = torch.tensor([[0, 0, 1, 1, 1]], device=torch_device)
        # Attention mask is all ones for this test case
        attention_mask = torch.ones_like(input_ids)

        # Expected position IDs: [0, 1, 0, 1, 2]
        expected_pos_ids = torch.tensor([[0, 1, 0, 1, 2]], device=torch_device)

        captured_kwargs = {}
        def hook(module, args, kwargs, output):
            captured_kwargs.update(kwargs)

        # Register hook to capture arguments passed to embeddings
        if version.parse(torch.__version__) >= version.parse("2.0"):
             handle = model.embeddings.register_forward_hook(hook, with_kwargs=True)
        else:
             # Fallback for older torch if needed, but FlexAttention requires new torch anyway
             handle = model.embeddings.register_forward_hook(hook, with_kwargs=True)

        model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids)
        handle.remove()

        self.assertIn("position_ids", captured_kwargs)
        self.assertTrue(torch.equal(captured_kwargs["position_ids"], expected_pos_ids))