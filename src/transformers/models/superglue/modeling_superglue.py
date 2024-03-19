# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""PyTorch SuperGlue model."""
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from transformers.models.superglue.configuration_superglue import SuperGlueConfig
from transformers import PreTrainedModel

from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC_ = "SuperGlueConfig"

_CHECKPOINT_FOR_DOC_ = "stevenbucaille/superglue"

SUPERGLUE_PRETRAINED_MODEL_ARCHIVE_LIST = ["stevenbucaille/superglue"]


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.einsum('bhnm,bdhm->bdhn', prob, value)
    return output, prob


def normalize_keypoints(
        keypoints: torch.Tensor,
        height: int,
        width: int
):
    """ Normalize keypoints locations based on image image_shape"""
    one = keypoints.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (keypoints - center[:, None, :]) / scaling[:, None, :]


def log_sinkhorn_iterations(
        Z: torch.Tensor,
        log_mu: torch.Tensor,
        log_nu: torch.Tensor,
        iters: int
) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


@dataclass
class ImageMatchingOutput(ModelOutput):
    """
    TODO documentation
    """

    image0_matches: torch.FloatTensor = None
    image1_matches: torch.FloatTensor = None
    image0_matching_scores: torch.FloatTensor = None
    image1_matching_scores: torch.FloatTensor = None


class SuperGlueMultiLayerPerceptron(nn.Module):
    def __init__(
            self,
            config: SuperGlueConfig,
            channels: List[int],
            do_batch_norm: bool = True,
    ):
        super().__init__()
        num_layers = len(channels)
        layers = []
        for i in range(1, num_layers):
            layers.append(
                nn.Conv1d(
                    channels[i - 1],
                    channels[i],
                    kernel_size=1,
                    bias=True
                )
            )
            if i < (num_layers - 1):
                if do_batch_norm:
                    layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        nn.init.constant_(self.layers[-1].bias, 0.0)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)


class SuperGlueKeypointEncoder(nn.Module):

    def __init__(
            self,
            config: SuperGlueConfig,
    ):
        super().__init__()
        self.layer_sizes = config.keypoint_encoder_sizes
        self.feature_dim = config.descriptor_dim
        self.encoder = SuperGlueMultiLayerPerceptron(
            config, [3] + self.layer_sizes + [self.feature_dim]
        )

    def forward(self, keypoints: Tensor, scores: Tensor) -> Tensor:
        keypoints = keypoints.transpose(1, 2)
        scores = scores.unsqueeze(1)
        inputs = torch.cat([keypoints, scores], dim=1)
        return self.encoder(inputs)


class SuperGlueMultiHeadAttention(nn.Module):
    def __init__(
            self,
            config: SuperGlueConfig
    ):
        super().__init__()
        self.feature_dim = config.descriptor_dim
        self.num_heads = config.num_heads
        self.dim = self.feature_dim // self.num_heads
        self.merge = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor
    ) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [
            layer(x).view(batch_dim, self.dim, self.num_heads, -1)
            for layer, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        output = self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))
        return output


class SuperGlueAttentionalPropagation(nn.Module):
    def __init__(
            self,
            config: SuperGlueConfig,
    ):
        super().__init__()
        self.descriptor_dim = config.descriptor_dim
        self.num_heads = config.num_heads
        self.attention = SuperGlueMultiHeadAttention(config)
        self.mlp = SuperGlueMultiLayerPerceptron(
            config, [self.descriptor_dim * 2, self.descriptor_dim * 2, self.descriptor_dim]
        )
        nn.init.constant_(self.mlp.layers[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attention(x, source, source)
        message = torch.cat([x, message], dim=1)
        message = self.mlp(message)
        return message


class SuperGlueAttentionalGNN(nn.Module):
    def __init__(
            self,
            config: SuperGlueConfig,
    ):
        super().__init__()
        self.descriptor_dim = config.descriptor_dim
        self.num_heads = config.num_heads
        self.layers_types = config.gnn_layers_types
        self.num_layers = len(self.layers_types)
        self.layers = nn.ModuleList(
            [
                SuperGlueAttentionalPropagation(
                    config,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, descriptors_0: torch.Tensor, descriptors_1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for gnn_layer, type in zip(self.layers, self.layers_types):
            if type == 'cross':
                source_0, source_1 = descriptors_1, descriptors_0
            else:  # if type == 'self':
                source_0, source_1 = descriptors_0, descriptors_1

            delta0 = gnn_layer(descriptors_0, source_0)
            delta1 = gnn_layer(descriptors_1, source_1)
            descriptors_0 = descriptors_0 + delta0
            descriptors_1 = descriptors_1 + delta1
        return descriptors_0, descriptors_1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """

    def __init__(
            self,
            config: SuperGlueConfig,
    ):
        super().__init__()

        self.keypoint_encoder = SuperGlueKeypointEncoder(config)
        self.gnn = SuperGlueAttentionalGNN(config)

        self.descriptor_dim = config.descriptor_dim
        self.final_proj = nn.Conv1d(self.descriptor_dim, self.descriptor_dim, kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.matching_threshold = config.matching_threshold

    def forward(
            self,
            keypoints_0: Tensor,
            scores_0: Tensor,
            descriptors_0: Tensor,
            keypoints_1: Tensor,
            scores_1: Tensor,
            descriptors_1: Tensor
    ):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        if keypoints_0.shape[1] == 0 or keypoints_1.shape[1] == 0:  # no keypoints
            shape0, shape1 = keypoints_0.shape[:-1], keypoints_1.shape[:-1]
            return (
                keypoints_0.new_full(shape0, -1, dtype=torch.int),
                keypoints_1.new_full(shape1, -1, dtype=torch.int),
                keypoints_0.new_zeros(shape0),
                keypoints_1.new_zeros(shape1)
            )

        # Keypoint MLP encoder.
        descriptors_0 = descriptors_0 + self.keypoint_encoder(keypoints_0, scores_0)
        descriptors_1 = descriptors_1 + self.keypoint_encoder(keypoints_1, scores_1)

        # Multi-layer Transformer network.
        descriptors_0, descriptors_1 = self.gnn(descriptors_0, descriptors_1)

        # Final MLP projection.
        projected_descriptors_0 = self.final_proj(descriptors_0)
        projected_descriptors_1 = self.final_proj(descriptors_1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', projected_descriptors_0, projected_descriptors_1)
        scores = scores / self.descriptor_dim ** .5

        # Run the optimal transport.
        scores = self.log_optimal_transport(
            scores,
            self.bin_score,
            iters=self.sinkhorn_iterations
        )

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = self.arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = self.arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        matching_scores_0 = torch.where(mutual0, max0.values.exp(), zero)
        matching_scores_1 = torch.where(mutual1, matching_scores_0.gather(1, indices1), zero)
        valid0 = mutual0 & (matching_scores_0 > self.matching_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)
        matches_0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        matches_1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return matches_0, matches_1, matching_scores_0, matching_scores_1

    @staticmethod
    def arange_like(x, dim: int):
        return x.new_ones(x.shape[dim]).cumsum(0) - 1


class SuperGluePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SuperGlueConfig
    base_model_prefix = "superglue"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


SUPERGLUE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SuperPointConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

# TODO
SUPERGLUE_INPUTS_DOCSTRING = r""" 
    """


class SuperGlueModel(SuperGluePreTrainedModel):
    #TODO documentation

    def __init__(self, config: SuperGlueConfig):
        super().__init__(config)

        self.keypoint_encoder = SuperGlueKeypointEncoder(config)
        self.gnn = SuperGlueAttentionalGNN(config)

        self.final_proj = nn.Conv1d(config.descriptor_dim, config.descriptor_dim, kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    @add_start_docstrings_to_model_forward(SUPERGLUE_INPUTS_DOCSTRING)
    def forward(
            self,
            image0_keypoints: Tensor = None,
            image0_scores: Tensor = None,
            image0_descriptors: Tensor = None,
            image1_keypoints: Tensor = None,
            image1_scores: Tensor = None,
            image1_descriptors: Tensor = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageMatchingOutput]:
        # TODO documentation example

        if image0_keypoints.shape[1] == 0 or image1_keypoints.shape[1] == 0:  # no keypoints
            shape0, shape1 = image0_keypoints.shape[:-1], image1_keypoints.shape[:-1]
            return (
                image0_keypoints.new_full(shape0, -1, dtype=torch.int),
                image1_keypoints.new_full(shape1, -1, dtype=torch.int),
                image0_keypoints.new_zeros(shape0),
                image1_keypoints.new_zeros(shape1)
            )

            # Keypoint MLP encoder.
        descriptors_0 = image0_descriptors + self.keypoint_encoder(image0_keypoints, image0_scores)
        descriptors_1 = image1_descriptors + self.keypoint_encoder(image1_keypoints, image1_scores)

        # Multi-layer Transformer network.
        descriptors_0, descriptors_1 = self.gnn(descriptors_0, descriptors_1)

        # Final MLP projection.
        projected_descriptors_0 = self.final_proj(descriptors_0)
        projected_descriptors_1 = self.final_proj(descriptors_1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', projected_descriptors_0, projected_descriptors_1)
        scores = scores / self.descriptor_dim ** .5

        # Run the optimal transport.
        scores = self.log_optimal_transport(
            scores,
            self.bin_score,
            iters=self.sinkhorn_iterations
        )

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = self.arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = self.arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        matching_scores_0 = torch.where(mutual0, max0.values.exp(), zero)
        matching_scores_1 = torch.where(mutual1, matching_scores_0.gather(1, indices1), zero)
        valid0 = mutual0 & (matching_scores_0 > self.config.matching_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)
        matches_0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        matches_1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        if not return_dict:
            return matches_0, matches_1, matching_scores_0, matching_scores_1

        return ImageMatchingOutput(
            image0_matches=matches_0,
            image1_matches=matches_1,
            image0_matching_scores=matching_scores_0,
            image1_matching_scores=matching_scores_1,
        )
