# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch LeViT model. """


import inspect
import unittest
from math import floor, ceil

from transformers import LevitConfig
from transformers.file_utils import cached_property, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import LevitForImageClassification, LevitModel
    from transformers.models.levit.modeling_levit import LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import AutoFeatureExtractor


class LevitConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "embed_dim"))
        self.parent.assertTrue(hasattr(config, "num_heads"))


class LevitModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size = 64,
        num_channels = 3,
        kernel_size = 3,
        stride = 2,
        padding = 1,
        patch_size = 16,
        embed_dim = [128, 256, 384],
        num_heads = [4, 6, 8],
        depth = [2, 3, 4],
        key_dim = [16, 16, 16],
        drop_path_rate = 0,
        mlp_ratio = [2, 2, 2],
        attention_ratio = [2, 2, 2],
        distillation = True,
        initializer_range=0.02,
        is_training=True,
        use_labels=True,
        num_labels=2,  # Check
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.key_dim = key_dim
        self.drop_path = drop_path_rate
        self.patch_size = patch_size
        self.attention_ratio = attention_ratio
        self.mlp_ratio = mlp_ratio
        self.distillation = distillation
        self.initializer_range = initializer_range
        self.down_ops = [
            ['Subsample', key_dim[0], embed_dim[0]// key_dim[0], 4, 2, 2],
            ['Subsample', key_dim[0], embed_dim[1]// key_dim[0], 4, 2, 2]
        ]
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.initializer_range = initializer_range

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self):
        return LevitConfig(
            image_size=self.image_size,
            num_channels = self.num_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding= self.padding,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads = self.num_heads,
            depth = self.depth,
            key_dim=self.key_dim,
            drop_path_rate=self.drop_path,
            mlp_ratio=self.mlp_ratio,
            attention_ratio=self.attention_ratio,
            distillation=self.distillation, 
            initializer_range=self.initializer_range,
            down_ops=self.down_ops
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = LevitModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        image_size = (self.image_size, self.image_size)
        height, width = image_size[0], image_size[1]
        for _ in range(4):
            height = floor(((height + 2 * self.padding - self.kernel_size) / self.stride) + 1)
            width = floor(((width + 2 * self.padding - self.kernel_size) / self.stride) + 1)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, ceil(height/4) * ceil(width/4), self.embed_dim[-1]))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = LevitForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class LevitModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Levit does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (LevitModel, LevitForImageClassification) if is_torch_available() else ()

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = LevitModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LevitConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        return

    @unittest.skip(reason="Levit does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Levit does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            for i in hidden_states:
                print(i.shape)

            expected_num_layers = sum(self.model_tester.depth) * 2 + len(self.model_tester.down_ops) * 2
            self.assertEqual(len(hidden_states), expected_num_layers)

            image_size = (self.model_tester.image_size, self.model_tester.image_size)
            height, width = image_size[0], image_size[1]
            for _ in range(4):
                height = floor(((height + 2 * self.model_tester.padding - self.model_tester.kernel_size) / self.model_tester.stride) + 1)
                width = floor(((width + 2 * self.model_tester.padding - self.model_tester.kernel_size) / self.model_tester.stride) + 1)
            # verify the first hidden states (first block)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [
                    height * width,
                    self.model_tester.embed_dim[0],
                ],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = LevitModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class LevitModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return AutoFeatureExtractor.from_pretrained(LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST[0])

    @slow
    def test_inference_image_classification_head(self):
        model = LevitForImageClassification.from_pretrained(LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST[0]).to(torch_device)

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-0.4981, 1.0718, 0.5807]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
