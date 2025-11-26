# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BERT model configuration"""

from collections import OrderedDict
from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder. If not
            provided, it is computed as `int((8 / 3) * hidden_size)` following the common SwiGLU sizing rule.
            For a hidden_size of 768, this defaults to 2048. This value controls the size of both the gate and
            value projections in the SwiGLU activation.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) for heads that keep the legacy BERT MLPs (for
            example, the masked LM head). If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
        ffn_activation (`str`, *optional*, defaults to `"swiglu"`):
            Feed-forward activation used inside the encoder. This flex-attention variant is implemented with SwiGLU
            feed-forward networks and only supports `"swiglu"` here.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
            if using Next Sentence Prediction (NSP) during pre-training.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. This modernized BERT variant only supports `"absolute"` position embeddings.
            Relative position embeddings (`"relative_key"`, `"relative_key_query"`) are not supported due to
            the use of flex attention.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. This modernized BERT variant only supports `False` (encoder-only).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import BertConfig, BertModel

    >>> # Initializing a BERT google-bert/bert-base-uncased style configuration
    >>> configuration = BertConfig()

    >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=None,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        pad_token_id=0,
        ffn_activation: str = "swiglu",
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # Validate encoder-only architecture - this variant does not support decoder mode
        if kwargs.get("is_decoder", False):
            raise ValueError(
                "This modernized BERT variant does not support decoder mode (is_decoder=True). "
                "This implementation is designed for encoder-only use cases."
            )
        if kwargs.get("add_cross_attention", False):
            raise ValueError(
                "This modernized BERT variant does not support cross-attention (add_cross_attention=True). "
                "This implementation is designed for encoder-only use cases."
            )

        # Validate position embedding type - only absolute is supported in this modernized variant
        if position_embedding_type != "absolute":
            raise ValueError(
                f"This modernized BERT variant only supports 'absolute' position embeddings, "
                f"got '{position_embedding_type}'. Relative position embeddings are not supported "
                f"with flex attention."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        if ffn_activation != "swiglu":
            raise ValueError("This modernized BERT variant only supports `ffn_activation='swiglu'`.")
        self.ffn_activation = ffn_activation
        self.intermediate_size = (
            int((8.0 / 3.0) * hidden_size) if intermediate_size is None else intermediate_size
        )
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self._attn_implementation = "flex_attention"


class BertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )


__all__ = ["BertConfig", "BertOnnxConfig"]
