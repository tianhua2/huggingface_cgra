# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""
Processor class for LayoutLMv3.
"""

import sys
import warnings
from typing import List, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, TextKwargs
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


class LayoutLMv3TextKwargs(TextKwargs, total=False):
    text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]]
    boxes: Optional[Union[List[List[int]], List[List[List[int]]]]]
    word_labels: Optional[Union[List[int], List[List[int]]]]


class LayoutLMv3ImagesKwargs(ImagesKwargs, total=False):
    apply_ocr: bool
    ocr_lang: Optional[str]
    tesseract_config: Optional[str]


class LayoutLMv3ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: LayoutLMv3TextKwargs
    images_kwargs: LayoutLMv3ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_length": False,
            "verbose": True,
        },
        "images_kwargs": {
            "apply_ocr": True,
        },
    }


class LayoutLMv3Processor(ProcessorMixin):
    r"""
    Constructs a LayoutLMv3 processor which combines a LayoutLMv3 image processor and a LayoutLMv3 tokenizer into a
    single processor.

    [`LayoutLMv3Processor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`LayoutLMv3ImageProcessor`] to resize and normalize document images, and optionally applies OCR to
    get words and normalized bounding boxes. These are then provided to [`LayoutLMv3Tokenizer`] or
    [`LayoutLMv3TokenizerFast`], which turns the words and bounding boxes into token-level `input_ids`,
    `attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
    into token-level `labels` for token classification tasks (such as FUNSD, CORD).

    Args:
        image_processor (`LayoutLMv3ImageProcessor`, *optional*):
            An instance of [`LayoutLMv3ImageProcessor`]. The image processor is a required input.
        tokenizer (`LayoutLMv3Tokenizer` or `LayoutLMv3TokenizerFast`, *optional*):
            An instance of [`LayoutLMv3Tokenizer`] or [`LayoutLMv3TokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LayoutLMv3ImageProcessor"
    tokenizer_class = ("LayoutLMv3Tokenizer", "LayoutLMv3TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        **kwargs: Unpack[LayoutLMv3ProcessorKwargs],
    ) -> BatchEncoding:
        """
        This method first forwards the `images` argument to [`~LayoutLMv3ImageProcessor.__call__`]. In case
        [`LayoutLMv3ImageProcessor`] was initialized with `apply_ocr` set to `True`, it passes the obtained words and
        bounding boxes along with the additional arguments to [`~LayoutLMv3Tokenizer.__call__`] and returns the output,
        together with resized and normalized `pixel_values`. In case [`LayoutLMv3ImageProcessor`] was initialized with
        `apply_ocr` set to `False`, it passes the words (`text`/``text_pair`) and `boxes` specified by the user along
        with the additional arguments to [`~LayoutLMv3Tokenizer.__call__`] and returns the output, together with
        resized and normalized `pixel_values`.

        Please refer to the docstring of the above two methods for more information.
        """
        output_kwargs = self._merge_kwargs(
            LayoutLMv3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_pair = output_kwargs["text_kwargs"].pop("text_pair", None)
        boxes = output_kwargs["text_kwargs"].pop("boxes", None)
        word_labels = output_kwargs["text_kwargs"].pop("word_labels", None)
        apply_ocr = output_kwargs["images_kwargs"].get("apply_ocr", self.image_processor.apply_ocr)

        # verify input
        if apply_ocr and (boxes is not None):
            raise ValueError(
                "You cannot provide bounding boxes if you initialized the image processor with apply_ocr set to True."
            )

        if apply_ocr and (word_labels is not None):
            raise ValueError(
                "You cannot provide word labels if you initialized the image processor with apply_ocr set to True."
            )

        # first, apply the image processor
        features = self.image_processor(images=images, **output_kwargs["images_kwargs"])

        # second, apply the tokenizer
        if text is not None and apply_ocr and text_pair is None:
            if isinstance(text, str):
                text = [text]  # add batch dimension (as the image processor always adds a batch dimension)
            text_pair = features["words"]

        if text is None:
            if not hasattr(features, "words"):
                raise ValueError("You need to provide `text` or set `apply_ocr` to `True`")
            text = features["words"]
        if boxes is None:
            if not hasattr(features, "boxes"):
                raise ValueError("You need to provide `boxes` or set `apply_ocr` to `True`")
            boxes = features["boxes"]

        encoded_inputs = self.tokenizer(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            **output_kwargs["text_kwargs"],
        )

        # add pixel values
        images = features.pop("pixel_values")
        if output_kwargs["text_kwargs"]["return_overflowing_tokens"]:
            images = self.get_overflowing_images(images, encoded_inputs["overflow_to_sample_mapping"])
        encoded_inputs["pixel_values"] = images

        return encoded_inputs

    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        # in case there's an overflow, ensure each `input_ids` sample is mapped to its corresponding image
        images_with_overflow = []
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])

        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )

        return images_with_overflow

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "attention_mask", "pixel_values"]

    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
