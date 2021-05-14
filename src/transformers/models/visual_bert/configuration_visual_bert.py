# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" VisualBERT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "gchhablani/visualbert-vqa": "https://huggingface.co/gchhablani/visualbert-vqa/resolve/main/config.json",
    "gchhablani/visualbert-vqa-pre": "https://huggingface.co/gchhablani/visualbert-vqa-pre/resolve/main/config.json",
    "gchhablani/visualbert-vqa-coco-pre": "https://huggingface.co/gchhablani/visualbert-vqa-coco-pre/resolve/main/config.json",
    "gchhablani/visualbert-vcr": "https://huggingface.co/gchhablani/visualbert-vcr/resolve/main/config.json",
    "gchhablani/visualbert-vcr-pre": "https://huggingface.co/gchhablani/visualbert-vcr-pre/resolve/main/config.json",
    "gchhablani/visualbert-vcr-coco-pre": "https://huggingface.co/gchhablani/visualbert-vcr-coco-pre/resolve/main/config.json",
    "gchhablani/visualbert-nlvr2": "https://huggingface.co/gchhablani/visualbert-nlvr2/resolve/main/config.json",
    "gchhablani/visualbert-nlvr2-pre": "https://huggingface.co/gchhablani/visualbert-nlvr2-pre/resolve/main/config.json",
    "gchhablani/visualbert-nlvr2-coco-pre": "https://huggingface.co/gchhablani/visualbert-nlvr2-coco-pre/resolve/main/config.json"
    # See all VisualBERT models at https://huggingface.co/models?filter=visual_bert
}


class VisualBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.VisualBertModel`. It is used
    to instantiate an VisualBERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the VisualBERT
    `visualbert-vqa-coco-pre <https://huggingface.co/gchhablani/visualbert-vqa-coco-pre>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the VisualBERT model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.VisualBertModel`. Vocabulary size of the
            model. Defines the different tokens that can be represented by the `inputs_ids` passed to the forward
            method of :class:`~transformers.VisualBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        visual_embedding_dim (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the visual embeddings to be passed to the model.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling
            :class:`~transformers.VisualBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        bypass_transformer (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should bypass the transformer for the visual embeddings. If set to `True`, the
            model directly concatenates the visual embeddings from :class:`~transformers.VisualBertEmbeddings` with
            text output from transformers, and then pass it to a self-attention layer.


        Example::

        >>> from transformers import VisualBertModel, VisualBertConfig

        >>> # Initializing a VisualBERT visualbert-vqa-coco-pre style configuration
        >>> configuration = VisualBertConfig.from_pretrained('visualbert-vqa-coco-pre')

        >>> # Initializing a model from the visualbert-vqa-coco-pre style configuration
        >>> model = VisualBertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "visual_bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        visual_embedding_dim=512,  # TO-DO: Need to check original visual embedding dim
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        bypass_transformer=False,
        # TO-DO: Check if the following parameters are needed and if yes, then whether they are to be documented.
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.visual_embedding_dim = visual_embedding_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.bypass_transformer = bypass_transformer
