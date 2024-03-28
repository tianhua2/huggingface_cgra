# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Gemma model. """
import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import AutoModelForCausalLM, AutoTokenizer, RecurrentGemmaConfig, is_torch_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import RecurrentGemmaForCausalLM, RecurrentGemmaModel


class RecurrentGemmaModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=64,
        is_training=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        width=32,
        mlp_expanded_width=128,
        num_hidden_layers=3,
        num_heads=4,
        lru_width=64,
        attention_window_size=16,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.width = width
        self.num_hidden_layers = num_hidden_layers
        self.mlp_expanded_width = mlp_expanded_width
        self.num_heads = num_heads
        self.lru_width = lru_width
        self.attention_window_size = attention_window_size
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    # Copied from tests.models.gemma.GemmaModelTester.prepare_config_and_inputs
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
        position_ids = torch.arange(self.seq_length)
        position_ids = torch.repeat_interleave(position_ids[None], self.batch_size, dim=0)

        return config, input_ids, token_type_ids, position_ids, sequence_labels, token_labels, choice_labels

    # Ignore copy
    def get_config(self):
        return RecurrentGemmaConfig(
            num_hidden_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            width=self.width,
            mlp_expanded_width=self.mlp_expanded_width,
            num_heads=self.num_heads,
            lru_width=self.lru_width,
            attention_window_size=self.attention_window_size,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            pad_token_id=self.pad_token_id,
        )

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model with Llama->Gemma
    def create_and_check_model(
        self, config, input_ids, token_type_ids, position_ids, sequence_labels, token_labels, choice_labels
    ):
        model = RecurrentGemmaModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, position_ids=position_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model_as_decoder with Llama->Gemma
    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = RecurrentGemmaModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, position_ids=position_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_for_causal_lm with Llama->Gemma
    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = RecurrentGemmaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, position_ids=position_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_decoder_model_past_large_inputs with Llama->Gemma
    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        position_ids,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = RecurrentGemmaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            position_ids=next_position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            position_ids=next_position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common with Llama->Gemma
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            position_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "position_ids": position_ids}
        return config, inputs_dict


@require_torch
class GemmaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (RecurrentGemmaModel, RecurrentGemmaForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (RecurrentGemmaForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": RecurrentGemmaModel,
            "text-generation": RecurrentGemmaForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    # Need to remove 0.9 in `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.6]

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        return True

    def setUp(self):
        self.model_tester = RecurrentGemmaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=RecurrentGemmaConfig, hidden_size=64)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    # def test_Gemma_sequence_classification_model(self):
    #     config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #     print(config)
    #     config.num_labels = 3
    #     input_ids = input_dict["input_ids"]
    #     attention_mask = input_ids.ne(1).to(torch_device)
    #     sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
    #     model = GemmaForSequenceClassification(config)
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
    #     self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))
    #
    # def test_Gemma_sequence_classification_model_for_single_label(self):
    #     config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #     config.num_labels = 3
    #     config.problem_type = "single_label_classification"
    #     input_ids = input_dict["input_ids"]
    #     attention_mask = input_ids.ne(1).to(torch_device)
    #     sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
    #     model = GemmaForSequenceClassification(config)
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
    #     self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # def test_Gemma_sequence_classification_model_for_multi_label(self):
    #     config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #     config.num_labels = 3
    #     config.problem_type = "multi_label_classification"
    #     input_ids = input_dict["input_ids"]
    #     attention_mask = input_ids.ne(1).to(torch_device)
    #     sequence_labels = ids_tensor(
    #         [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
    #     ).to(torch.float)
    #     model = GemmaForSequenceClassification(config)
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
    #     self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # @unittest.skip("TODO @gante fix this for Llama")
    # @parameterized.expand([(1, False), (1, True), (4, False)])
    # def test_new_cache_format(self, num_beams, do_sample):
    #     pass
    #
    # @unittest.skip("Gemma buffers include complex numbers, which breaks this test")
    # def test_save_load_fast_init_from_base(self):
    #     pass
    #
    # @unittest.skip("Gemma uses GQA on all models so the KV cache is a non standard format")
    # def test_past_key_values_format(self):
    #     pass
    #
    # @require_flash_attn
    # @require_torch_gpu
    # @pytest.mark.flash_attn_test
    # @slow
    # def test_flash_attn_2_generate_padding_right(self):
    #     import torch
    #
    #     for model_class in self.all_generative_model_classes:
    #         config, _ = self.model_tester.prepare_config_and_inputs_for_common()
    #         model = model_class(config)
    #
    #         with tempfile.TemporaryDirectory() as tmpdirname:
    #             model.save_pretrained(tmpdirname)
    #             model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(
    #                 torch_device
    #             )
    #
    #             dummy_input = torch.LongTensor([[0, 2, 3, 4], [0, 2, 3, 4]]).to(torch_device)
    #             dummy_attention_mask = torch.LongTensor([[1, 1, 1, 1], [1, 1, 1, 0]]).to(torch_device)
    #
    #             model.generate(dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=1, do_sample=False)
    #
    #             model = model_class.from_pretrained(
    #                 tmpdirname,
    #                 torch_dtype=torch.float16,
    #                 attn_implementation="flash_attention_2",
    #                 low_cpu_mem_usage=True,
    #             ).to(torch_device)
    #
    #             with self.assertRaises(ValueError):
    #                 _ = model.generate(
    #                     dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=1, do_sample=False
    #                 )
    #
    # @require_flash_attn
    # @require_torch_gpu
    # @pytest.mark.flash_attn_test
    # @slow
    # def test_flash_attn_2_generate_use_cache(self):
    #     import torch
    #
    #     max_new_tokens = 30
    #
    #     for model_class in self.all_generative_model_classes:
    #         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #
    #         dummy_input = inputs_dict[model_class.main_input_name]
    #         if dummy_input.dtype in [torch.float32, torch.bfloat16]:
    #             dummy_input = dummy_input.to(torch.float16)
    #
    #         # make sure that all models have enough positions for generation
    #         if hasattr(config, "max_position_embeddings"):
    #             config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1
    #
    #         model = model_class(config)
    #
    #         with tempfile.TemporaryDirectory() as tmpdirname:
    #             model.save_pretrained(tmpdirname)
    #
    #             dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
    #             # NOTE: Gemma apparently does not support right padding + use_cache with FA2.
    #             dummy_attention_mask[:, -1] = 1
    #
    #             model = model_class.from_pretrained(
    #                 tmpdirname,
    #                 torch_dtype=torch.float16,
    #                 attn_implementation="flash_attention_2",
    #                 low_cpu_mem_usage=True,
    #             ).to(torch_device)
    #
    #             # Just test that a large cache works as expected
    #             _ = model.generate(
    #                 dummy_input,
    #                 attention_mask=dummy_attention_mask,
    #                 max_new_tokens=max_new_tokens,
    #                 do_sample=False,
    #                 use_cache=True,
    #             )
    #
    # @require_flash_attn
    # @require_torch_gpu
    # @pytest.mark.flash_attn_test
    # @slow
    # def test_flash_attn_2_inference_padding_right(self):
    #     self.skipTest("Gemma flash attention does not support right padding")
    #
    # @require_torch_sdpa
    # @require_torch_gpu
    # @slow
    # def test_sdpa_equivalence(self):
    #     for model_class in self.all_model_classes:
    #         if not model_class._supports_sdpa:
    #             return
    #
    #         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #         model = model_class(config)
    #
    #         with tempfile.TemporaryDirectory() as tmpdirname:
    #             model.save_pretrained(tmpdirname)
    #             model_sdpa = model_class.from_pretrained(
    #                 tmpdirname, torch_dtype=torch.float16, attn_implementation="sdpa"
    #             )
    #             model_sdpa.to(torch_device)
    #
    #             model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="eager")
    #             model.to(torch_device)
    #
    #             dummy_input = inputs_dict[model_class.main_input_name]
    #             dummy_input = dummy_input.to(torch_device)
    #             outputs = model(dummy_input, output_hidden_states=True)
    #             outputs_sdpa = model_sdpa(dummy_input, output_hidden_states=True)
    #
    #             logits = outputs.hidden_states[-1]
    #             logits_sdpa = outputs_sdpa.hidden_states[-1]
    #
    #             # gemma sdpa needs a high tolerance
    #             assert torch.allclose(logits_sdpa, logits, atol=3e-3)
    #
    # @require_flash_attn
    # @require_torch_gpu
    # @pytest.mark.flash_attn_test
    # @slow
    # def test_flash_attn_2_equivalence(self):
    #     for model_class in self.all_model_classes:
    #         if not model_class._supports_flash_attn_2:
    #             return
    #
    #         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #         model = model_class(config)
    #
    #         with tempfile.TemporaryDirectory() as tmpdirname:
    #             model.save_pretrained(tmpdirname)
    #             model_fa = model_class.from_pretrained(
    #                 tmpdirname, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
    #             )
    #             model_fa.to(torch_device)
    #
    #             model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="eager")
    #             model.to(torch_device)
    #
    #             dummy_input = inputs_dict[model_class.main_input_name]
    #             dummy_input = dummy_input.to(torch_device)
    #             outputs = model(dummy_input, output_hidden_states=True)
    #             outputs_fa = model_fa(dummy_input, output_hidden_states=True)
    #
    #             logits = outputs.hidden_states[-1]
    #             logits_fa = outputs_fa.hidden_states[-1]
    #
    #             # gemma flash attention 2 needs a high tolerance
    #             assert torch.allclose(logits_fa, logits, atol=3e-3)
    #

@require_torch_gpu
@slow
@require_read_token
class GemmaIntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]

    def test_model_2b_fp32(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_2b_fp16(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_2b_fp16_static_cache(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )

        model.generation_config.cache_implementation = "static"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_2b_bf16(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Khichdi",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_2b_eager(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I am looking for some information on the ",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_torch_sdpa
    def test_model_2b_sdpa(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Khichdi",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @pytest.mark.flash_attn_test
    @require_flash_attn
    def test_model_2b_flash_attn(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_bitsandbytes
    def test_model_2b_4bit(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project and I need to make a 3d model of a house. I have been using",
            "Hi today I'd like to share with you my experience with the new wattpad wattpad wattpad wattpad wattpad wattpad wattpad",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, load_in_4bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @unittest.skip("The test will not fit our CI runners")
    def test_model_7b_fp32(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            "Hello my name is ***** ***** I will be assisting you today. I am sorry to hear about your issue. I will",
            "Hi,\n\nI have a problem with my 2005 1.6 16",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_7b_fp16(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            """Hello I am doing a project on a 1999 4.0L 4x4. I""",
            "Hi today I am going to show you how to make a simple and easy to make a DIY 3D",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_7b_bf16(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            """Hello I am doing a project on a 1991 240sx and I am trying to find""",
            "Hi today I am going to show you how to make a very simple and easy to make a very simple and",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_7b_fp16_static_cache(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            """Hello I am doing a project on a 1999 4.0L 4x4. I""",
            "Hi today I am going to show you how to make a simple and easy to make a DIY 3D",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )

        model.generation_config.cache_implementation = "static"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_bitsandbytes
    def test_model_7b_4bit(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project for my school and I am trying to make a program that will take a number and then",
            """Hi today I am going to talk about the new update for the game called "The new update" and I""",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, load_in_4bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)
