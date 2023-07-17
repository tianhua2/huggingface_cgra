# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and MosaicML NLP team.
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
""" Mpt configuration"""
import copy
from typing import TYPE_CHECKING, Optional, Union


if TYPE_CHECKING:
    pass

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mosaicml/mpt-7b": "https://huggingface.co/mosaicml/mpt-7b/resolve/main/config.json",
}


class MptAttentionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [*MptAttention*] class. It is used to instantiate
    attention layers according to the specified arguments, defining the layers architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MPT
    [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b) architecture.

    Configuration objects inherit from [*PretrainedConfig*] and can be used to control the model outputs. Read the
    documentation from [*PretrainedConfig*] for more information.

    Args:
        attn_type (str):
            type of attention to use. Options: multihead_attention, multiquery_attention
        attn_pdrop (float):
            The dropout probability for the attention layers. attn_impl (str): The attention implementation to use. One
            of 'torch', 'flash', or 'triton'.
        qk_ln (bool):
            Whether to apply layer normalization to the queries and keys in the attention layer.
        clip_qkv (Optional[float]):
            If not None, clip the queries, keys, and values in the attention layer to this value.
        softmax_scale (Optional[float]):
            If not None, scale the softmax in the attention layer by this value. If None, use the default scale of
            `1/sqrt(d_keys)`.
        prefix_lm (Optional[bool]):
            Whether the model should operate as a Prefix LM. This requires passing an extra *prefix_mask* argument
            which indicates which tokens belong to the prefix. Tokens in the prefix can attend to one another
            bi-directionally. Tokens outside the prefix use causal attention.
        attn_uses_sequence_id (Optional[bool]):
            Whether to restrict attention to tokens that have the same sequence_id. When the model is in *train* mode,
            this requires passing an extra *sequence_id* argument which indicates which sub-sequence each token belongs
            to. Defaults to `False` meaning any provided *sequence_id* will be ignored.
        alibi (bool):
            Whether to use the alibi bias instead of position embeddings. alibi_bias_max (int): The maximum value of
            the alibi bias.
    """

    def __init__(
        self,
        attn_type="multihead_attention",
        attn_pdrop=0,
        attn_impl="triton",
        normalise_query_key=False,
        clip_query_key_value=None,
        softmax_scale=None,  # TODO rename
        prefix_lm=False,  # TODO what is this
        attn_uses_sequence_id=False,
        alibi=False,
        alibi_bias_max=8,
        **kwargs,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.attn_pdrop = attn_pdrop
        self.attn_impl = attn_impl
        self.normalise_query_key = normalise_query_key
        self.clip_query_key_value = clip_query_key_value
        self.softmax_scale = softmax_scale
        self.prefix_lm = prefix_lm
        self.attn_uses_sequence_id = attn_uses_sequence_id
        self.alibi = alibi
        self.alibi_bias_max = alibi_bias_max


class MptIntializerConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [*MptAttention*] class. It is used to instantiate
    attention layers according to the specified arguments, defining the layers architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MPT
    [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b) architecture.

    Configuration objects inherit from [*PretrainedConfig*] and can be used to control the model outputs. Read the
    documentation from [*PretrainedConfig*] for more information.

    Args:
        init_config.name:
            The parameter initialization scheme to use. Options: 'default_', 'baseline_',
                'kaiming_uniform_', 'kaiming_normal_', 'neox_init_', 'small_init_', 'xavier_uniform_', or
                'xavier_normal_'. These mimic the parameter initialization methods in PyTorch.
            init_div_is_residual (Union[int, float, str, bool]):
                Value to divide initial weights by if `module._is_residual` is True.
            emb_init_std (Optional[float]):
                The standard deviation of the normal distribution used to initialize the embedding layer.
            emb_init_uniform_lim (Optional[Union[Tuple[float,float], float]]):
                The lower and upper limits of the uniform distribution used to initialize the embedding layer. Mutually
                exclusive with `emb_init_std`.
            init_std (float):
                The standard deviation of the normal distribution used to initialize the model, if using the baseline_
                parameter initialization scheme.
            init_gain (float):
                The gain to use for parameter initialization with kaiming or xavier initialization schemes.
            fan_mode (str): The fan mode to use for parameter initialization with kaiming initialization
                schemes.
            init_nonlinearity (str):
                The nonlinearity to use for parameter initialization with kaiming initialization schemes. --- See
                llmfoundry.models.utils.param_init_fns.py for info on other param init config options
    """

    def __init__(
        self,
        name="kaiming_normal_",
        fan_mode="fan_in",
        init_nonlinearity="relu",
        init_div_is_residual=True,
        emb_init_std=None,
        emb_init_uniform_lim=None,
        init_std=None,
        init_gain=0.0,
        **kwargs,
    ):
        super().__init__()

        self.name = name
        self.fan_mode = fan_mode
        self.init_nonlinearity = init_nonlinearity
        self.init_div_is_residual = init_div_is_residual
        self.emb_init_std = emb_init_std
        self.emb_init_uniform_lim = emb_init_uniform_lim
        self.init_std = init_std
        self.init_gain = init_gain


class MptConfig(PretrainedConfig):
    model_type = "mpt"
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        expansion_ratio: int = 4,
        max_seq_len: int = 2048,
        vocab_size: int = 50368,
        num_key_value_heads: int = 16,
        resid_pdrop: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        emb_pdrop: float = 0.0,
        learned_pos_emb: bool = True,
        attn_config: MptAttentionConfig = None,
        init_device: str = "cpu",
        logit_scale: Optional[Union[float, str]] = None,
        no_bias: bool = False,
        verbose: int = 0,
        embedding_fraction: float = 1.0,
        norm_type: str = "low_precision_layernorm",
        use_cache: bool = False,
        init_config: MptIntializerConfig = None,
        initializer_range=0.02,
        **kwargs,
    ):
        """
        This is the configuration class to store the configuration of a [`MptModel`]. It is used to instantiate a Mpt
        model according to the specified arguments, defining the model architecture. Instantiating a configuration with
        the defaults will yield a similar configuration to the Mpt architecture
        [bigscience/mpt](https://huggingface.co/bigscience/mpt).

        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.


        Args:
            vocab_size (`int`, *optional*, defaults to 250880):
                Vocabulary size of the Mpt model. Defines the maximum number of different tokens that can be
                represented by the `inputs_ids` passed when calling [`MptModel`]. Check [this
                discussion](https://huggingface.co/bigscience/mpt/discussions/120#633d28389addb8530b406c2a) on how the
                `vocab_size` has been defined.
            d_model (`int`, *optional*, defaults to 64):
                Dimensionality of the embeddings and hidden states.
            n_layer (`int`, *optional*, defaults to 2):
                Number of hidden layers in the Transformer encoder.
            n_head (`int`, *optional*, defaults to 8):
                Number of attention heads for each attention layer in the Transformer encoder.
            layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
                The epsilon to use in the layer normalization layers.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
                If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
            hidden_dropout (`float`, *optional*, defaults to 0.1):
                Dropout rate of the dropout function on the bias dropout.
            attention_dropout (`float`, *optional*, defaults to 0.1):
                Dropout rate applied to the attention probs
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models).
            pretraining_tp (`int`, *optional*, defaults to `1`):
                Experimental feature. Tensor parallelism rank used during pretraining with Megatron. Please refer to
                [this document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This
                value is necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
                issue](https://github.com/pytorch/pytorch/issues/76232). Note also that this is enabled only when
                `slow_but_exact=True`.

            expansion_ratio (int):
                The ratio of the up/down scale in the MLP.
            max_seq_len (int):
                The maximum sequence length of the model.
            resid_pdrop (float):
                The dropout probability applied to the attention output before combining with residual.
            emb_pdrop (float):
                The dropout probability for the embedding layer.
            learned_pos_emb (bool):
                Whether to use learned positional embeddings
            attn_config (Dict):  A dictionary used to configure the model's attention module:

            init_device (str):
                The device to use for parameter initialization. logit_scale (Optional[Union[float, str]]): If not None,
                scale the logits by this value.
            no_bias (bool):
                Whether to use bias in all layers.
            verbose (int):
                The verbosity level. 0 is silent.
            embedding_fraction (float):
                The fraction to scale the gradients of the embedding layer by.
            norm_type (str):
                choose type of norm to use
            multiquery_attention (bool):
                Whether to use multiquery attention implementation.
            use_cache (bool):
                Whether or not the model should return the last key/values attentions
            init_config (Dict):
                A dictionary used to configure the model initialization.

        Example:

            ```python
            >>> from transformers import MptConfig, MptModel

            >>> # Initializing a Mpt configuration
            >>> configuration = MptConfig()

            >>> # Initializing a model (with random weights) from the configuration
            >>> model = MptModel(configuration)

            >>> # Accessing the model configuration
            >>> configuration = model.config
            ```"""
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.num_key_value_heads = num_key_value_heads

        if attn_config is None:
            self.attn_config = MptAttentionConfig()
        elif isinstance(attn_config, dict):
            self.attn_config = MptAttentionConfig(**attn_config)
        elif isinstance(attn_config, MptAttentionConfig):
            self.attn_config = attn_config
        else:
            raise ValueError(
                f"`attn_config` has to be either a `MptAttentionConfig` or a dictionary. Received: {attn_config}"
            )

        if init_config is None:
            self.init_config = MptIntializerConfig()
        elif isinstance(init_config, dict):
            self.init_config = MptIntializerConfig(**init_config)
        elif isinstance(init_config, MptIntializerConfig):
            self.init_config = init_config
        else:
            raise ValueError(
                f"`init_config` has to be either a `MptIntializerConfig` or a dictionary. Received: {init_config}"
            )

        super().__init__(**kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["attn_config"] = (
            self.attn_config.to_dict() if not isinstance(self.attn_config, dict) else self.attn_config
        )
        output["init_config"] = (
            self.init_config.to_dict() if not isinstance(self.init_config, dict) else self.init_config
        )
        output["model_type"] = self.__class__.model_type
        return output
