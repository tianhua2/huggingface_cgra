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
"""Convert Jukebox checkpoints"""

import argparse
import json
import os
from pathlib import Path

import torch

import requests
from transformers import JukeboxConfig, JukeboxModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


PREFIX = "https://openaipublic.azureedge.net/jukebox/models/"
MODEL_MAPPING = {
    "jukebox-1b-lyrics": [
        "5b/vqvae.pth.tar",
        "5b/prior_level_0.pth.tar",
        "5b/prior_level_1.pth.tar",
        "1b_lyrics/prior_level_2.pth.tar",
    ],
    "jukebox-5b-lyrics": [
        "5b/vqvae.pth.tar",
        "5b/prior_level_0.pth.tar",
        "5b/prior_level_1.pth.tar",
        "5b_lyrics/prior_level_2.pth.tar",
    ],
}


def replace_key(key):
    if key.endswith(".model.1.bias") and len(key.split(".")) > 10:
        key = key.replace(".model.1.bias", ".conv1d_1.bias")
    elif key.endswith(".model.1.weight") and len(key.split(".")) > 10:
        key = key.replace(".model.1.weight", ".conv1d_1.weight")
    elif key.endswith(".model.3.bias") and len(key.split(".")) > 10:
        key = key.replace(".model.3.bias", ".conv1d_2.bias")
    elif key.endswith(".model.3.weight") and len(key.split(".")) > 10:
        key = key.replace(".model.3.weight", ".conv1d_2.weight")

    if "prime_prior" in key:
        key = key.replace("prime_prior", "lyric_encoder")

    if key.endswith("k"):  # replace vqvae.X.k with vqvae.X.codebook
        return key.replace(".k", ".codebook")
    if "y_emb." in key:
        return key.replace("y_emb.", "metadata_embedding.")
    if "prime_state_ln" in key:
        return key.replace("prime_state_ln", "lyric_encoder.final_layer_norm")
    if ".ln" in key:
        return key.replace(".ln", ".layer_norm")
    if "_ln" in key:
        return key.replace("_ln", "_layer_norm")

    if "prime_state_proj" in key:
        return key.replace("prime_state_proj", "lyric_encoder.proj_in")
    if "prime_x_out" in key:
        return key.replace("prime_x_out", "lyric_encoder.lm_head")
    if "prior.x_out" in key:
        return key.replace("x_out", "fc_proj_out")
    if "x_emb" in key:
        return key.replace("x_emb", "embed_tokens")
    return key


def fix_jukebox_keys(state_dict, model_state_dict, key_prefix, mapping):
    new_dict = {}
    import re

    re_encoder_block_conv_in = re.compile("encoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).(bias|weight)")
    re_encoder_block_resnet = re.compile(
        "encoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).model.(\d*).model.(\d*).(bias|weight)"
    )
    re_encoder_block_proj_out = re.compile("encoders.(\d*).level_blocks.(\d*).model.(\d*).(bias|weight)")

    re_decoder_block_conv_out = re.compile("decoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).(bias|weight)")
    re_decoder_block_resnet = re.compile(
        "decoders.(\d*).level_blocks.(\d*).model.(\d*).(\d).model.(\d*).model.(\d*).(bias|weight)"
    )
    re_decoder_block_proj_in = re.compile("decoders.(\d*).level_blocks.(\d*).model.(\d*).(bias|weight)")

    re_prior_cond_conv_out = re.compile("conditioner_blocks.(\d*).cond.model.(\d*).(\d).(bias|weight)")
    re_prior_cond_resnet = re.compile(
        "conditioner_blocks.(\d*).cond.model.(\d*).(\d).model.(\d*).model.(\d*).(bias|weight)"
    )
    re_prior_cond_proj_in = re.compile("conditioner_blocks.(\d*).cond.model.(\d*).(bias|weight)")

    for original_key, value in state_dict.items():

        # rename vqvae.encoder keys
        if re_encoder_block_conv_in.fullmatch(original_key):
            regex_match = re_encoder_block_conv_in.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[2]) * 2 + int(groups[3])
            re_new_key = f"encoders.{groups[0]}.level_blocks.{groups[1]}.downsample_block.{block_index}.{groups[-1]}"
            key = re_encoder_block_conv_in.sub(re_new_key, original_key)

        elif re_encoder_block_resnet.fullmatch(original_key):
            regex_match = re_encoder_block_resnet.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[2]) * 2 + int(groups[3])
            conv_index = {"1": 1, "3": 2}[groups[-2]]
            prefix = f"encoders.{groups[0]}.level_blocks.{groups[1]}.downsample_block.{block_index}."
            resnet_block = f"resnet_block.{groups[-3]}.conv1d_{conv_index}.{groups[-1]}"
            re_new_key = prefix + resnet_block
            key = re_encoder_block_resnet.sub(re_new_key, original_key)

        elif re_encoder_block_proj_out.fullmatch(original_key):
            regex_match = re_encoder_block_proj_out.match(original_key)
            groups = regex_match.groups()
            re_new_key = f"encoders.{groups[0]}.level_blocks.{groups[1]}.proj_out.{groups[-1]}"
            key = re_encoder_block_proj_out.sub(re_new_key, original_key)

        # rename vqvae.decoder keys
        elif re_decoder_block_conv_out.fullmatch(original_key):
            regex_match = re_decoder_block_conv_out.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[2]) * 2 + int(groups[3]) - 2
            re_new_key = f"decoders.{groups[0]}.level_blocks.{groups[1]}.upsample_block.{block_index}.{groups[-1]}"
            key = re_decoder_block_conv_out.sub(re_new_key, original_key)

        elif re_decoder_block_resnet.fullmatch(original_key):
            regex_match = re_decoder_block_resnet.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[2]) * 2 + int(groups[3]) - 2
            conv_index = {"1": 1, "3": 2}[groups[-2]]
            prefix = f"decoders.{groups[0]}.level_blocks.{groups[1]}.upsample_block.{block_index}."
            resnet_block = f"resnet_block.{groups[-3]}.conv1d_{conv_index}.{groups[-1]}"
            re_new_key = prefix + resnet_block
            key = re_decoder_block_resnet.sub(re_new_key, original_key)

        elif re_decoder_block_proj_in.fullmatch(original_key):
            regex_match = re_decoder_block_proj_in.match(original_key)
            groups = regex_match.groups()
            re_new_key = f"decoders.{groups[0]}.level_blocks.{groups[1]}.proj_in.{groups[-1]}"
            key = re_decoder_block_proj_in.sub(re_new_key, original_key)

        # rename prior cond.model to upsampler.upsample_block and resnet
        elif re_prior_cond_conv_out.fullmatch(original_key):
            regex_match = re_prior_cond_conv_out.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[1]) * 2 + int(groups[2]) - 2
            re_new_key = f"conditioner_blocks.{groups[0]}.upsampler.upsample_block.{block_index}.{groups[-1]}"
            key = re_prior_cond_conv_out.sub(re_new_key, original_key)

        elif re_prior_cond_resnet.fullmatch(original_key):
            regex_match = re_prior_cond_resnet.match(original_key)
            groups = regex_match.groups()
            block_index = int(groups[1]) * 2 + int(groups[2]) - 2
            conv_index = {"1": 1, "3": 2}[groups[-2]]
            prefix = f"conditioner_blocks.{groups[0]}.upsampler.upsample_block.{block_index}."
            resnet_block = f"resnet_block.{groups[-3]}.conv1d_{conv_index}.{groups[-1]}"
            re_new_key = prefix + resnet_block
            key = re_prior_cond_resnet.sub(re_new_key, original_key)

        elif re_prior_cond_proj_in.fullmatch(original_key):
            regex_match = re_prior_cond_proj_in.match(original_key)
            groups = regex_match.groups()
            re_new_key = f"conditioner_blocks.{groups[0]}.upsampler.proj_in.{groups[-1]}"
            key = re_prior_cond_proj_in.sub(re_new_key, original_key)

        # keep original key
        else:
            key = original_key

        key = replace_key(key)

        if f"{key_prefix}.{key}" not in model_state_dict or key is None:
            print(f"failed converting {original_key} to {key}, does not match")

        # handle missmatched shape
        elif value.shape != model_state_dict[f"{key_prefix}.{key}"].shape:
            val = model_state_dict[f"{key_prefix}.{key}"]
            print(f"{original_key}-> {key} : \nshape {val.shape} and { value.shape}, do not match")
            key = original_key

        mapping[key] = original_key
        new_dict[key] = value

    return new_dict


@torch.no_grad()
def convert_openai_checkpoint(model_name=None, pytorch_dump_folder_path=None):
    """
    Copy/paste/tweak model's weights to our Jukebox structure.
    """
    for file in MODEL_MAPPING[model_name]:
        if not os.path.isfile(f"{pytorch_dump_folder_path}/{file.split('/')[-1]}"):
            r = requests.get(f"{PREFIX}{file}", allow_redirects=True)
            os.makedirs(f"{pytorch_dump_folder_path}/", exist_ok=True)
            open(f"{pytorch_dump_folder_path}/{file.split('/')[-1]}", "wb").write(r.content)

    model_to_convert = MODEL_MAPPING[model_name.split("/")[-1]]

    # config = JukeboxConfig.from_pretrained("openai/" + model_name)
    # to convert the 5b lyric token model, use : or "openai/jukebox-5b-lyrics"
    # config = JukeboxConfig(
    #     timing_dims=128
    #     prior_attn_order=[10, 2, 2],
    #     prior_blocks=128,
    #     prime_n_vocab=80,
    #     nb_relevant_lyric_tokens=[512, 0, 0],
    #     prior_n_heads=[8, 1, 1],
    #     prior_n_ctx=[8192, 8192, 8192],
    #     prime_width=[1280, 128, 128],
    #     prior_width=[4800, 1920, 1920],
    #     single_enc_dec=[False, False, False],
    #     timing_dims=128,
    #     vqvae_width=64,
    #     metadata_dims=[(120, 4111), (120, 4111), (120, 4111)],
    #     min_duration=23.8,
    #     sample_length= 1058304,
    #     prior_depth=[79, 72, 72],
    #     max_nb_genres=1,
    # )
    config = JukeboxConfig(sample_length=1058304)
    model = JukeboxModel(config)

    weight_dict = []
    mapping = {}
    for i, dict_name in enumerate(model_to_convert):
        old_dic = torch.load(f"{pytorch_dump_folder_path}/{dict_name.split('/')[-1]}")["model"]

        new_dic = {}
        for k in old_dic.keys():
            if k.endswith(".b"):
                new_dic[k.replace("b", "bias")] = old_dic[k]
            elif k.endswith(".w"):
                new_dic[k.replace("w", "weight")] = old_dic[k]
            elif "level_2" not in dict_name and "cond.model." in k:
                new_dic[k.replace(".blocks.", ".model.")] = old_dic[k]
            else:
                new_dic[k] = old_dic[k]

        key_prefix = "vqvae" if i == 0 else f"priors.{i-1}"
        new_dic = fix_jukebox_keys(new_dic, model.state_dict(), key_prefix, mapping)
        weight_dict.append(new_dic)

    vqvae_state_dict = weight_dict.pop(0)
    model.vqvae.load_state_dict(vqvae_state_dict)
    for i in range(len(weight_dict)):
        model.priors[i].load_state_dict(weight_dict[i])

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    with open(f"{pytorch_dump_folder_path}/mapping.json", "w") as txtfile:
        json.dump(mapping, txtfile, sep="\n")

    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    return weight_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="jukebox-1b-lyrics",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="jukebox-1b-lyrics-converted",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    args = parser.parse_args()
    convert_openai_checkpoint(args.model_name, args.pytorch_dump_folder_path)
