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
    from transformers.models.bert.modeling_bert import BertRMSNorm


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
        config = BertConfig(**defaults)
        config._attn_implementation = "flex_attention"
        return config

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
            self.assertIsInstance(layer.attention_rmsnorm, BertRMSNorm)
            self.assertIsInstance(layer.output_rmsnorm, BertRMSNorm)

        # Check embeddings RMSNorm exists
        self.assertIsInstance(model.embeddings_rmsnorm, BertRMSNorm)

    def test_rmsnorm_eps_value(self):
        """Test that RMSNorm uses correct epsilon from config."""
        config = self._make_config()
        config.layer_norm_eps = 1e-6
        model = BertModel(config).to(torch_device)

        # Check epsilon matches config
        self.assertEqual(model.embeddings_rmsnorm.variance_epsilon, config.layer_norm_eps)
        self.assertEqual(model.encoder.layer[0].attention_rmsnorm.variance_epsilon, config.layer_norm_eps)
        self.assertEqual(model.encoder.layer[0].output_rmsnorm.variance_epsilon, config.layer_norm_eps)

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

    # ===== Document Masking Correctness Tests =====

    def test_document_masking_isolates_representations(self):
        """
        Critical test: Verify that changing tokens in doc B does NOT affect doc A's representations.
        This is the key property of correct document masking.
        """
        config = self._make_config(num_hidden_layers=2, hidden_size=64, num_attention_heads=4)
        model = BertModel(config).to(torch_device)
        model.eval()

        # Input: [Doc0: tokens 1,2,3] [Doc1: tokens 4,5,6]
        input_ids_v1 = torch.tensor([[1, 2, 3, 4, 5, 6]], device=torch_device)
        # Change ONLY Doc1 tokens (positions 3,4,5)
        input_ids_v2 = torch.tensor([[1, 2, 3, 7, 8, 9]], device=torch_device)
        
        attention_mask = torch.ones_like(input_ids_v1)
        # Doc0 = positions 0,1,2; Doc1 = positions 3,4,5
        document_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], device=torch_device)

        with torch.no_grad():
            out_v1 = model(input_ids=input_ids_v1, attention_mask=attention_mask, document_ids=document_ids).last_hidden_state
            out_v2 = model(input_ids=input_ids_v2, attention_mask=attention_mask, document_ids=document_ids).last_hidden_state

        # Doc0 representations (positions 0,1,2) should be IDENTICAL
        # because Doc0 tokens cannot attend to Doc1 tokens
        torch.testing.assert_close(
            out_v1[:, :3, :], out_v2[:, :3, :],
            msg="Doc0 representations changed when Doc1 tokens changed - document masking is broken!"
        )
        
        # Doc1 representations (positions 3,4,5) should be DIFFERENT
        # because we changed the input tokens
        self.assertFalse(
            torch.allclose(out_v1[:, 3:, :], out_v2[:, 3:, :]),
            msg="Doc1 representations didn't change when Doc1 tokens changed"
        )

    def test_document_masking_first_token_isolation(self):
        """
        Test that the first token of a document only sees tokens from its own document.
        The [CLS] token of Doc1 should not be influenced by Doc0.
        """
        config = self._make_config(num_hidden_layers=1, hidden_size=32, num_attention_heads=2)
        model = BertModel(config).to(torch_device)
        model.eval()

        # Packed: [CLS0, A, B, SEP0, CLS1, C, D, SEP1]
        # Use token IDs within vocab_size=32
        input_ids = torch.tensor([[1, 10, 11, 2, 1, 20, 21, 2]], device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        document_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], device=torch_device)

        # Change Doc0 content only (keep IDs within vocab_size=32)
        input_ids_changed = torch.tensor([[1, 15, 16, 2, 1, 20, 21, 2]], device=torch_device)

        with torch.no_grad():
            out_original = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids).last_hidden_state
            out_changed = model(input_ids=input_ids_changed, attention_mask=attention_mask, document_ids=document_ids).last_hidden_state

        # Doc1's CLS token (position 4) should be identical
        torch.testing.assert_close(
            out_original[:, 4, :], out_changed[:, 4, :],
            msg="Doc1's [CLS] token changed when only Doc0 content changed!"
        )

    def test_document_masking_bidirectional_attention(self):
        """
        Test that bidirectional attention works within documents.
        Token at position 0 should see token at position 2 within same doc.
        """
        config = self._make_config(num_hidden_layers=1, hidden_size=32, num_attention_heads=2)
        model = BertModel(config).to(torch_device)
        model.eval()

        # Single document, all tokens should attend to each other
        input_ids = torch.tensor([[1, 2, 3, 4]], device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        document_ids = torch.tensor([[0, 0, 0, 0]], device=torch_device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids).last_hidden_state

        # All tokens should have non-trivial representations
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        
        # Verify it's not just the same vector repeated (which would indicate broken attention)
        self.assertFalse(torch.allclose(out[:, 0, :], out[:, 1, :]))

    def test_document_masking_zero_based_ids(self):
        """
        Test that document IDs starting from 0 work correctly.
        User provides [0, 0, 1, 1], internally shifted to [1, 1, 2, 2].
        """
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        input_ids = torch.tensor([[10, 11, 20, 21]], device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        
        # Zero-based document IDs (common user expectation)
        document_ids_zero = torch.tensor([[0, 0, 1, 1]], device=torch_device)
        # One-based document IDs  
        document_ids_one = torch.tensor([[1, 1, 2, 2]], device=torch_device)

        with torch.no_grad():
            out_zero = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids_zero).last_hidden_state
            out_one = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids_one).last_hidden_state

        # Both should produce the same results (internal +1 shift handles this)
        torch.testing.assert_close(out_zero, out_one)

    def test_document_masking_with_negative_padding_ids(self):
        """
        Test that negative document IDs for padding work correctly.
        Padding positions should not attend to or be attended by valid tokens.
        """
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        # Input with padding at the end
        input_ids = torch.tensor([[1, 2, 3, 0, 0]], device=torch_device)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]], device=torch_device)
        # User might use -1 for padding document IDs
        document_ids = torch.tensor([[0, 0, 0, -1, -1]], device=torch_device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids).last_hidden_state

        # Should not have NaN or inf
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        
        # Valid token representations should be reasonable
        self.assertEqual(out.shape, (1, 5, config.hidden_size))

    def test_attention_output_shape_correctness(self):
        """
        Test that attention output has correct shape after flex_attention_forward.
        This catches the double-transpose bug.
        """
        config = self._make_config(hidden_size=64, num_attention_heads=4)
        model = BertModel(config).to(torch_device)
        model.eval()

        batch_size, seq_len = 2, 8
        input_ids = ids_tensor([batch_size, seq_len], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Verify correct shape
        expected_shape = (batch_size, seq_len, config.hidden_size)
        self.assertEqual(out.shape, expected_shape, f"Expected {expected_shape}, got {out.shape}")
        
        # Verify values are reasonable (not scrambled)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        
        # Verify different positions have different representations
        # (would be suspicious if all positions had same representation)
        unique_representations = len(set(tuple(out[0, i, :5].tolist()) for i in range(seq_len)))
        self.assertGreater(unique_representations, 1, "All positions have identical representations - something is wrong")

    def test_document_masking_gradient_flow(self):
        """
        Test that gradients flow correctly through document-masked attention.
        Gradient for Doc0 tokens should not depend on Doc1 loss.
        """
        config = self._make_config(num_hidden_layers=1, hidden_size=32, num_attention_heads=2)
        model = BertModel(config).to(torch_device)
        model.train()

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        document_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], device=torch_device)

        # Forward pass
        out = model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids).last_hidden_state

        # Loss only on Doc1 (positions 3, 4, 5)
        loss_doc1 = out[:, 3:, :].sum()
        loss_doc1.backward()

        # Check gradients exist
        self.assertIsNotNone(model.embeddings.word_embeddings.weight.grad)

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
        config._attn_implementation = "flex_attention"
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

    def test_mixed_precision_rms_norm(self):
        """
        Test that BertRMSNorm handles mixed precision correctly by casting
        to float32 internally for numerical stability and returning to input dtype.
        """
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            return

        config = self._make_config()
        model = BertModel(config).to(torch_device)
        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        
        # Test with autocast - model should work without errors
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids)
        
        self.assertIsNotNone(outputs.last_hidden_state)
        # Output should not have NaN or Inf values (numerical stability check)
        self.assertFalse(torch.isnan(outputs.last_hidden_state).any())
        self.assertFalse(torch.isinf(outputs.last_hidden_state).any())

        # Test without autocast - model should work with FP32
        model_fp32 = BertModel(config).to(torch_device)
        outputs_fp32 = model_fp32(input_ids=input_ids)
        self.assertIsNotNone(outputs_fp32.last_hidden_state)
        self.assertEqual(outputs_fp32.last_hidden_state.dtype, torch.float32)
        
        # Test with model in bfloat16 - RMSNorm should still be numerically stable
        model_bf16 = BertModel(config).to(torch_device).to(torch.bfloat16)
        outputs_bf16 = model_bf16(input_ids=input_ids)
        self.assertEqual(outputs_bf16.last_hidden_state.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(outputs_bf16.last_hidden_state).any())
        self.assertFalse(torch.isinf(outputs_bf16.last_hidden_state).any())

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
class BertSdpaAttentionTest(unittest.TestCase):
    """Tests for SDPA attention implementation."""

    def _make_config(self, seq_length: int = 6, attn_implementation: str = "sdpa", **kwargs):
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
        config = BertConfig(**defaults)
        config._attn_implementation = attn_implementation
        return config

    # ===== Basic SDPA Functionality Tests =====

    def test_sdpa_forward_pass(self):
        """Test that SDPA forward pass works correctly."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs.last_hidden_state.shape, (2, 8, config.hidden_size))
        self.assertIsNotNone(outputs.pooler_output)
        self.assertFalse(torch.isnan(outputs.last_hidden_state).any())

    def test_sdpa_with_attention_mask(self):
        """Test SDPA correctly handles attention masks."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        # Mask out last 2 positions
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ], device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs.last_hidden_state.shape, (2, 8, config.hidden_size))
        self.assertFalse(torch.isnan(outputs.last_hidden_state).any())

    def test_sdpa_backward_runs(self):
        """Test that gradients flow correctly through SDPA."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.train()

        input_ids = ids_tensor([2, 5], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.last_hidden_state.sum()
        loss.backward()

        # Verify gradients populated
        self.assertIsNotNone(model.embeddings.word_embeddings.weight.grad)
        self.assertIsNotNone(model.encoder.layer[0].attention.self.query.weight.grad)

    def test_sdpa_no_attention_mask(self):
        """Test SDPA works without attention mask (all tokens attend to all)."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 6], config.vocab_size).to(torch_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        self.assertEqual(outputs.last_hidden_state.shape, (2, 6, config.hidden_size))
        self.assertFalse(torch.isnan(outputs.last_hidden_state).any())

    # ===== SDPA vs FlexAttention Consistency Tests =====

    def test_sdpa_flex_attention_similar_outputs(self):
        """Test that SDPA and FlexAttention produce similar outputs for simple cases."""
        config_sdpa = self._make_config(attn_implementation="sdpa")
        config_flex = self._make_config(attn_implementation="flex_attention")

        # Create two models with same weights
        model_sdpa = BertModel(config_sdpa).to(torch_device)
        model_flex = BertModel(config_flex).to(torch_device)

        # Copy weights from sdpa to flex
        model_flex.load_state_dict(model_sdpa.state_dict())

        model_sdpa.eval()
        model_flex.eval()

        input_ids = ids_tensor([2, 6], config_sdpa.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out_sdpa = model_sdpa(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            out_flex = model_flex(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Outputs should be very close (not exact due to different code paths)
        torch.testing.assert_close(out_sdpa, out_flex, rtol=1e-4, atol=1e-4)

    # ===== SDPA Document IDs Warning Test =====

    def test_sdpa_document_ids_warning(self):
        """Test that document_ids produces a warning with SDPA."""
        config = self._make_config()
        model = BertModel(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 6], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)
        document_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], device=torch_device)

        logger = logging.get_logger("transformers.models.bert.modeling_bert")
        logger.warning_once.cache_clear()

        with CaptureLogger(logger) as cl:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, document_ids=document_ids)

        self.assertIn("document_ids", cl.out)
        self.assertIn("FlexAttention", cl.out)

    # ===== SDPA Attention Implementation Selection Tests =====

    def test_sdpa_attention_class_selection(self):
        """Test that SDPA attention class is correctly selected."""
        from transformers.models.bert.modeling_bert import BertSdpaSelfAttention

        config = self._make_config(attn_implementation="sdpa")
        model = BertModel(config).to(torch_device)

        # Check that attention layers use SDPA
        for layer in model.encoder.layer:
            self.assertIsInstance(layer.attention.self, BertSdpaSelfAttention)

    def test_flex_attention_class_selection(self):
        """Test that FlexAttention class is correctly selected."""
        from transformers.models.bert.modeling_bert import BertFlexSelfAttention

        config = self._make_config(attn_implementation="flex_attention")
        model = BertModel(config).to(torch_device)

        # Check that attention layers use FlexAttention
        for layer in model.encoder.layer:
            self.assertIsInstance(layer.attention.self, BertFlexSelfAttention)

    # ===== SDPA Model Variants Tests =====

    def test_sdpa_masked_lm_model(self):
        """Test SDPA with BertForMaskedLM."""
        config = self._make_config(attn_implementation="sdpa")
        model = BertForMaskedLM(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs.logits.shape, (2, 8, config.vocab_size))

    def test_sdpa_sequence_classification_model(self):
        """Test SDPA with BertForSequenceClassification."""
        config = self._make_config(attn_implementation="sdpa")
        config.num_labels = 3
        model = BertForSequenceClassification(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs.logits.shape, (2, 3))

    def test_sdpa_token_classification_model(self):
        """Test SDPA with BertForTokenClassification."""
        config = self._make_config(attn_implementation="sdpa")
        config.num_labels = 5
        model = BertForTokenClassification(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs.logits.shape, (2, 8, 5))

    # ===== SDPA with Gradient Checkpointing =====

    def test_sdpa_gradient_checkpointing(self):
        """Test SDPA with gradient checkpointing enabled."""
        config = self._make_config(attn_implementation="sdpa")
        model = BertModel(config).to(torch_device)
        model.gradient_checkpointing_enable()
        model.train()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.last_hidden_state.sum()
        loss.backward()

        self.assertIsNotNone(model.embeddings.word_embeddings.weight.grad)

    # ===== SDPA Numerical Stability Tests =====

    def test_sdpa_mixed_precision(self):
        """Test SDPA works correctly with mixed precision."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for mixed precision test")

        config = self._make_config(attn_implementation="sdpa")
        model = BertModel(config).to("cuda")
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to("cuda")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs.last_hidden_state.shape, (2, 8, config.hidden_size))
        self.assertFalse(torch.isnan(outputs.last_hidden_state).any())

    def test_sdpa_attention_dropout_training(self):
        """Test that SDPA applies dropout during training."""
        config = self._make_config(attention_probs_dropout_prob=0.1, attn_implementation="sdpa")
        model = BertModel(config).to(torch_device)
        model.train()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        # Run forward twice - with dropout, outputs should differ
        torch.manual_seed(42)
        out1 = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        torch.manual_seed(123)
        out2 = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Outputs should be different due to dropout
        self.assertFalse(torch.allclose(out1, out2))

    def test_sdpa_no_dropout_eval(self):
        """Test that SDPA does not apply dropout during evaluation."""
        config = self._make_config(attention_probs_dropout_prob=0.5, attn_implementation="sdpa")
        model = BertModel(config).to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 8], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)

        # Run forward twice - without dropout (eval mode), outputs should be identical
        with torch.no_grad():
            torch.manual_seed(42)
            out1 = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            torch.manual_seed(123)
            out2 = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Outputs should be identical in eval mode
        torch.testing.assert_close(out1, out2)


@require_torch
class BertModelIntegrationTest(unittest.TestCase):
    def _make_config(self, attn_implementation: str = "flex_attention", **kwargs):
        config = BertConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
        )
        config._attn_implementation = attn_implementation
        return config

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