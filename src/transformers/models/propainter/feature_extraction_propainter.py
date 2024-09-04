# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the S-Lab License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/sczhou/ProPainter/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Feature extractor class for ProPainter."""

import warnings

from ...utils import logging
from .image_processing_propainter import ProPainterImageProcessor


logger = logging.get_logger(__name__)


class ProPainterFeatureExtractor(ProPainterImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class ProPainterFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use ProPainterImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)
