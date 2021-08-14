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
""" Testing suite for the PyTorch FNet model. """


import unittest
from typing import Dict, List, Tuple

from transformers import FNetConfig, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, slow, torch_device

from transformers.models.fnet.modeling_fnet import FNetBasicFourierTransform, _scipy_available
from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        FNetForMaskedLM,
        FNetForMultipleChoice,
        FNetForNextSentencePrediction,
        FNetForPreTraining,
        FNetForQuestionAnswering,
        FNetForSequenceClassification,
        FNetForTokenClassification,
        FNetModel,
    )
    from transformers.models.fnet.modeling_fnet import FNET_PRETRAINED_MODEL_ARCHIVE_LIST


# Override ConfigTester
class FNetConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        if self.has_text_modality:
            self.parent.assertTrue(hasattr(config, "vocab_size"))
        self.parent.assertTrue(hasattr(config, "hidden_size"))
        self.parent.assertTrue(hasattr(config, "num_hidden_layers"))


class FNetModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
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
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

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

        return config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return FNetConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            tpu_short_seq_length=self.seq_length,
        )

    def create_and_check_fourier_transform(self, config):
        hidden_states = floats_tensor([self.batch_size, self.seq_length, config.hidden_size])
        transform = FNetBasicFourierTransform(config)
        fftn_output = transform(hidden_states)

        config.use_tpu_fourier_optimizations = True
        if _scipy_available:
            transform = FNetBasicFourierTransform(config)
            dft_output = transform(hidden_states)

        config.max_position_embeddings = 4097
        transform = FNetBasicFourierTransform(config)
        fft_output = transform(hidden_states)

        if _scipy_available:
            self.parent.assertTrue(torch.allclose(fftn_output[0][0], dft_output[0][0], atol=1e-4))
            self.parent.assertTrue(torch.allclose(fft_output[0][0], dft_output[0][0], atol=1e-4))
        self.parent.assertTrue(torch.allclose(fftn_output[0][0], fft_output[0][0], atol=1e-4))

    def create_and_check_model(self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels):
        model = FNetModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_pretraining(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        model = FNetForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            labels=token_labels,
            next_sentence_label=sequence_labels,
        )
        self.parent.assertEqual(result.prediction_logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertEqual(result.seq_relationship_logits.shape, (self.batch_size, 2))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        model = FNetForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_next_sentence_prediction(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        model = FNetForNextSentencePrediction(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            next_sentence_label=sequence_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, 2))

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        model = FNetForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = FNetForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = FNetForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = FNetForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(
            multiple_choice_inputs_ids,
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
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids}
        return config, inputs_dict


@require_torch
class FNetModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            FNetModel,
            FNetForPreTraining,
            FNetForMaskedLM,
            FNetForNextSentencePrediction,
            FNetForMultipleChoice,
            FNetForQuestionAnswering,
            FNetForSequenceClassification,
            FNetForTokenClassification,
        )
        if is_torch_available()
        else ()
    )

    # Skip Tests
    test_pruning = False
    test_torchscript = False
    test_head_masking = False
    test_pruning = False

    # special case for ForPreTraining model
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
                inputs_dict["next_sentence_label"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
        return inputs_dict

    # Overriden Tests
    def test_attention_outputs(self):
        pass

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (List, Tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=f"Tuple and dict output are not equal. Difference: {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`: {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}.",
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            # tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            # dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            # check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        hidden_states = outputs.hidden_states[0]

        hidden_states.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)

    def setUp(self):
        self.model_tester = FNetModelTester(self)
        self.config_tester = FNetConfigTester(self, config_class=FNetConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in FNET_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = FNetModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class FNetModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = FNetForMaskedLM.from_pretrained("gchhablani/fnet-base")
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        vocab_size = 32000

        expected_shape = torch.Size((1, 6, vocab_size))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[-1.7770, -7.7386, -7.5002], [-3.4747, -8.5963, -7.7719], [-3.2070, -9.0645, -8.3344]]]
        )

        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
