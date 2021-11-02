# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import itertools
import os
import subprocess

from parameterized import parameterized
from transformers import is_torch_available
from transformers.testing_utils import (
    ExtendSysPath,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    require_deepspeed,
    require_torch_gpu,
    slow,
)
from transformers.trainer_utils import set_seed


tests_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
root_dir = os.path.dirname(tests_dir)
with ExtendSysPath(tests_dir):
    from test_trainer import TrainerIntegrationCommon  # noqa

    if is_torch_available():
        from test_trainer import RegressionModelConfig, RegressionPreTrainedModel, get_regression_trainer  # noqa


set_seed(42)

T5_SMALL = "t5-small"

ALBERT_TINY = "hf-internal-testing/tiny-albert"
BART_TINY = "sshleifer/bart-tiny-random"
BERT_TINY = "hf-internal-testing/tiny-bert"
BIGBIRD_PEGASUS_TINY = "hf-internal-testing/tiny-random-bigbird_pegasus"
BIG_BIRD_TINY = "hf-internal-testing/tiny-random-big_bird"
BLENDERBOT_TINY = "hf-internal-testing/tiny-random-blenderbot"
DEBERTA_TINY = "hf-internal-testing/tiny-deberta"
DEBERTA_V2_TINY = "hf-internal-testing/tiny-random-deberta-v2"
DISTILBERT_TINY = "sshleifer/tiny-distilbert-base-cased"
ELECTRA_TINY = "hf-internal-testing/tiny-electra"
FLAUBERT_TINY = "hf-internal-testing/tiny-random-flaubert"
FSMT_TINY = "stas/tiny-wmt19-en-de"
FUNNEL_TINY = "hf-internal-testing/tiny-random-funnel"
GPT2_TINY = "sshleifer/tiny-gpt2"
GPTJ_TINY = "hf-internal-testing/tiny-random-gptj"
GPT_NEO_TINY = "hf-internal-testing/tiny-random-gpt_neo"
LAYOUTLM_TINY = "hf-internal-testing/tiny-layoutlm"
LED_TINY = "hf-internal-testing/tiny-random-led"
LONGFORMER_TINY = "hf-internal-testing/tiny-random-longformer"
M2M_100_TINY = "hf-internal-testing/tiny-random-m2m_100"
MARIAN_TINY = "sshleifer/tiny-marian-en-de"
MBART_TINY = "sshleifer/tiny-mbart"
MOBILEBERT_TINY = "hf-internal-testing/tiny-random-mobilebert"
MPNET_TINY = "hf-internal-testing/tiny-random-mpnet"
PEGASUS_TINY = "stas/pegasus-cnn_dailymail-tiny-random"
PROPHETNET_TINY = "hf-internal-testing/tiny-random-prophetnet"
ROBERTA_TINY = "sshleifer/tiny-distilroberta-base"
SQUEEZEBERT_TINY = "hf-internal-testing/tiny-random-squeezebert"
T5_TINY = "patrickvonplaten/t5-tiny-random"
T5_V1_TINY = "hf-internal-testing/tiny-random-t5-v1.1"
VIT_TINY = "hf-internal-testing/tiny-random-vit"
XLM_ROBERTA_TINY = "hf-internal-testing/tiny-xlm-roberta"
XLNET_TINY = "sshleifer/tiny-xlnet-base-cased"


# Issues and missing model files
CAMEMBERT_TINY = "hf-internal-testing/tiny-random-camembert"
CLIP_TINY = "hf-internal-testing/tiny-random-clip"
CONVBERT_TINY = "hf-internal-testing/tiny-random-convbert"
CTRL_TINY = "hf-internal-testing/tiny-random-ctrl"
HUBERT_TINY = "hf-internal-testing/tiny-random-hubert"
IBERT_TINY = "hf-internal-testing/tiny-random-ibert"
LAYOUTLMV2_TINY = "hf-internal-testing/tiny-random-layoutlmv2"
LXMERT_TINY = "hf-internal-testing/tiny-random-lxmert"
MT5_TINY = "hf-internal-testing/tiny-random-mt5"
OPENAI_GPT_TINY = "hf-internal-testing/tiny-random-openai-gpt"
RAG_TINY = "hf-internal-testing/tiny-random-rag"
REFORMER_TINY = "hf-internal-testing/tiny-random-reformer"
SPEECH_TO_TEXT_TINY = "hf-internal-testing/tiny-random-speech_to_text"
TRANSFO_XL_TINY = "hf-internal-testing/tiny-random-transfo-xl"

# *** Reactive mode ***
# models with low usage, unstable API, things about to change - do nothing about the following until someone runs into a problem
TAPAS_TINY = "hf-internal-testing/tiny-random-tapas"
# additional notes on tapas
# 1. requires torch_scatter - skip if it's not installed?
# 2. "Table must be of type pd.DataFrame" failure


# XXX: m2m_100 still needs work under z3 (works with z2) after PR is merged and this branch is re-based
# PYTHONPATH=src deepspeed  --master_port 6666 --num_nodes 1 --num_gpus 2 examples/pytorch/translation/run_translation.py --train_file tests/fixtures/tests_samples/wmt_en_ro/train.json --source_lang en --target_lang ro --model_name_or_path hf-internal-testing/tiny-random-m2m_100 --do_train --max_train_samples 4 --per_device_train_batch_size 2 --num_train_epochs 1 --fp16 --report_to none --overwrite_output_dir --deepspeed tests/deepspeed/ds_config_zero3.json --output_dir /tmp/tmpi4k4wz8s --save_steps 1
# if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
# AttributeError: 'Parameter' object has no attribute 'ds_status'

# TODO: to add:
#

# XXX:

#
# ** No official examples to test with:
# - "dpr" DPR_TINY ="hf-internal-testing/tiny-random-dpr" examples/research_projects/rag-end2end-retriever/
# - "rag" research_projects
# - "luke" Entities classes - no plan to make such example
# - "lxmert" doesn't work with run_qa.py
# - "clip" nothing under pytorch examples - XXX: Suraj is working on adding some - check by end of Sep
# - "speech_to_text", nothing under pytorch examples


def get_launcher(distributed=False):
    # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
    # - it won't be able to handle that
    # 2. for now testing with just 2 gpus max (since some quality tests may give different
    # results with mode gpus because we use very little data)
    num_gpus = min(2, get_gpu_count()) if distributed else 1
    return f"deepspeed --num_nodes 1 --num_gpus {num_gpus}".split()


def make_task_cmds():
    data_dir_fixtures = f"{tests_dir}/fixtures"
    data_dir_samples = f"{data_dir_fixtures}/tests_samples"
    data_dir_wmt = f"{data_dir_samples}/wmt_en_ro"
    data_dir_xsum = f"{data_dir_samples}/xsum"
    args_main = """
        --do_train
        --max_train_samples 4
        --per_device_train_batch_size 2
        --num_train_epochs 1
        --fp16
        --report_to none
        --overwrite_output_dir
        """.split()

    # XXX: try to cover as many models as possible once (it's enough to run on one task per model)
    # but need a tiny model for each
    #
    # should have T5_TINY, etc. global var defined
    tasks2models = dict(
        trans=[
            "bart",
            "fsmt",
            "marian",
            "mbart",
            "t5",
            "t5_v1",
            "m2m_100",
            # "mt5", missing model files
        ],
        sum=[
            "pegasus",
        ],
        clm=[
            "gpt2",
            "xlm-roberta",
            "gpt_neo",
            # "camembert", missing model files
            "big_bird",
            "gptj",
            "blenderbot",
            "bigbird_pegasus",
            "prophetnet",
        ],
        mlm=[
            "electra",
            "distilbert",
            "albert",
            "layoutlm",
            "deberta-v2",
            "funnel",
            "flaubert",
            # "reformer", # multiple issues with either mlm/qa/clas
        ],
        qa=[
            "roberta",
            "deberta",
            "longformer",
            "mobilebert",
            "mpnet",
            "led",
            "squeezebert",
            # "convbert", # missing tokenizer files
            # "layoutlmv2", missing model files
        ],
        clas=[
            "bert",
            "xlnet",
            # "hubert", # missing tokenizer files
            # "ibert", # multiple issues with either mlm/qa/clas
            # "transfo-xl", # tokenizer issues as ctrl
            # "ctrl", # tokenizer issues
            # "openai-gpt", missing model files
            # "tapas", multiple issues
        ],
        img_clas=[
            "vit",
        ],
    )

    scripts_dir = f"{root_dir}/examples/pytorch"

    tasks = dict(
        trans=f"""
        {scripts_dir}/translation/run_translation.py
        --train_file {data_dir_wmt}/train.json
        --source_lang en
        --target_lang ro
        """,
        sum=f"""
        {scripts_dir}/summarization/run_summarization.py
        --train_file {data_dir_xsum}/sample.json
        --max_source_length 12
        --max_target_length 12
        """,
        clm=f"""
        {scripts_dir}/language-modeling/run_clm.py
        --train_file {data_dir_fixtures}/sample_text.txt
        --block_size 8
        """,
        mlm=f"""
        {scripts_dir}/language-modeling/run_mlm.py
        --train_file {data_dir_fixtures}/sample_text.txt
        """,
        qa=f"""
        {scripts_dir}/question-answering/run_qa.py
        --train_file {data_dir_samples}/SQUAD/sample.json
        """,
        clas=f"""
        {scripts_dir}/text-classification/run_glue.py
        --train_file {data_dir_samples}/MRPC/train.csv
        --max_seq_length 12
        --task_name MRPC
        """,
        img_clas=f"""
        {scripts_dir}/image-classification/run_image_classification.py
            --dataset_name hf-internal-testing/cats_vs_dogs_sample
            --remove_unused_columns False
            --max_steps 10
            --image_size=30
        """,
    )

    launcher = get_launcher(distributed=True)

    cmds = {}
    for task, args in tasks.items():
        args = args.split()
        for model in tasks2models[task]:
            model_name = globals()[f"{model.upper().replace('-', '_')}_TINY"]
            args_model = f"--model_name_or_path {model_name}".split()
            cmds[f"{task}_{model}"] = launcher + args + args_model + args_main

            # # generation special case
            # if task == "gen":
            #     launcher = f"deepspeed --num_nodes 1 --num_gpus 1".split()
            #     args_model += f"--model_type {model}".split()
            #     cmds[f"{task}_{model}"] = launcher + args + args_model
            # else:

    return cmds


task_cmds = make_task_cmds()

ZERO2 = "zero2"
ZERO3 = "zero3"
stages = [ZERO2, ZERO3]


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


# Cartesian-product of zero stages with models to test
params = list(itertools.product(stages, task_cmds.keys()))


@slow
@require_deepspeed
@require_torch_gpu
class TestDeepSpeedModelZoo(TestCasePlus):
    """This class is for testing via an external script - can do multiple gpus"""

    def get_task_cmd(self, task, stage):
        # return a ready to run train cmd
        if task not in task_cmds:
            raise ValueError(f"don't know of task {task}, have {task_cmds.keys()}")

        cmd = task_cmds[task]
        args_ds = f"--deepspeed {self.test_file_dir_str}/ds_config_{stage}.json".split()

        output_dir = self.get_auto_remove_tmp_dir()
        args_out = f"--output_dir {output_dir}".split()

        cmd += args_ds + args_out

        return cmd, output_dir

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_zero_to_fp32(self, stage, task):
        # testing the ability to do a run followed by recovery of full fp32 weights

        cmd, output_dir = self.get_task_cmd(task, stage)

        # 1. generate the checkpoint
        cmd += "--save_steps 1".split()
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] + cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        # 2. test that the fp32 weights get reconsolidated
        chkpt_dir = f"{output_dir}/checkpoint-1"
        recovered_model_path = f"{chkpt_dir}/out.bin"
        cmd = f"{chkpt_dir}/zero_to_fp32.py {chkpt_dir} {recovered_model_path}"
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        subprocess.check_call(cmd, shell=True)
        assert os.path.exists(recovered_model_path), f"{recovered_model_path} was not found"

        # possibly could also test that the resulting saved model is usable but given that we use
        # random models we won't know if it's any good
