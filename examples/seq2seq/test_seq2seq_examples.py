import argparse
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import pytorch_lightning as pl
import torch

import lightning_base
from convert_pl_checkpoint_to_hf import convert_pl_to_hf
from distillation import distill_main, evaluate_checkpoint
from finetune import SummarizationModule, main
from run_eval import generate_summaries_or_translations, run_generate
from run_eval_search import run_search
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers.hf_api import HfApi
from transformers.testing_utils import CaptureStderr, CaptureStdout, require_multigpu, require_torch_and_cuda, slow
from utils import label_smoothed_nll_loss, lmap, load_json


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
CUDA_AVAILABLE = torch.cuda.is_available()
CHEAP_ARGS = {
    "max_tokens_per_batch": None,
    "supervise_forward": True,
    "normalize_hidden": True,
    "label_smoothing": 0.2,
    "eval_max_gen_length": None,
    "eval_beams": 1,
    "val_metric": "loss",
    "save_top_k": 1,
    "adafactor": True,
    "early_stopping_patience": 2,
    "logger_name": "default",
    "length_penalty": 0.5,
    "cache_dir": "",
    "task": "summarization",
    "num_workers": 2,
    "alpha_hid": 0,
    "freeze_embeds": True,
    "enc_only": False,
    "tgt_suffix": "",
    "resume_from_checkpoint": None,
    "sortish_sampler": True,
    "student_decoder_layers": 1,
    "val_check_interval": 1.0,
    "output_dir": "",
    "fp16": False,  # TODO(SS): set this to CUDA_AVAILABLE if ci installs apex or start using native amp
    "no_teacher": False,
    "fp16_opt_level": "O1",
    "gpus": 1 if CUDA_AVAILABLE else 0,
    "n_tpu_cores": 0,
    "max_grad_norm": 1.0,
    "do_train": True,
    "do_predict": True,
    "accumulate_grad_batches": 1,
    "server_ip": "",
    "server_port": "",
    "seed": 42,
    "model_name_or_path": "sshleifer/bart-tiny-random",
    "config_name": "",
    "tokenizer_name": "facebook/bart-large",
    "do_lower_case": False,
    "learning_rate": 0.3,
    "lr_scheduler": "linear",
    "weight_decay": 0.0,
    "adam_epsilon": 1e-08,
    "warmup_steps": 0,
    "max_epochs": 1,
    "train_batch_size": 2,
    "eval_batch_size": 2,
    "max_source_length": 12,
    "max_target_length": 12,
    "val_max_target_length": 12,
    "test_max_target_length": 12,
    "fast_dev_run": False,
    "no_cache": False,
    "n_train": -1,
    "n_val": -1,
    "n_test": -1,
    "student_encoder_layers": 1,
    "alpha_loss_encoder": 0.0,
    "freeze_encoder": False,
    "auto_scale_batch_size": False,
}


def _dump_articles(path: Path, articles: list):
    content = "\n".join(articles)
    Path(path).open("w").writelines(content)


ARTICLES = [" Sam ate lunch today.", "Sams lunch ingredients."]
SUMMARIES = ["A very interesting story about what I ate for lunch.", "Avocado, celery, turkey, coffee"]
T5_TINY = "patrickvonplaten/t5-tiny-random"
BART_TINY = "sshleifer/bart-tiny-random"
MBART_TINY = "sshleifer/tiny-mbart"
MARIAN_TINY = "sshleifer/tiny-marian-en-de"
FSMT_TINY = "stas/tiny-wmt19-en-de"


stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logging.disable(logging.CRITICAL)  # remove noisy download output from tracebacks


def make_test_data_dir(**kwargs):
    tmp_dir = Path(tempfile.mkdtemp(**kwargs))
    for split in ["train", "val", "test"]:
        _dump_articles((tmp_dir / f"{split}.source"), ARTICLES)
        _dump_articles((tmp_dir / f"{split}.target"), SUMMARIES)
    return tmp_dir


class TestSummarizationDistiller(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)  # remove noisy download output from tracebacks
        return cls

    @slow
    @require_torch_and_cuda
    def test_hub_configs(self):
        """I put require_torch_and_cuda cause I only want this to run with self-scheduled."""

        model_list = HfApi().model_list()
        org = "sshleifer"
        model_ids = [x.modelId for x in model_list if x.modelId.startswith(org)]
        allowed_to_be_broken = ["sshleifer/blenderbot-3B", "sshleifer/blenderbot-90M"]
        failures = []
        for m in model_ids:
            if m in allowed_to_be_broken:
                continue
            try:
                AutoConfig.from_pretrained(m)
            except Exception:
                failures.append(m)
        assert not failures, f"The following models could not be loaded through AutoConfig: {failures}"

    @require_multigpu
    def test_multigpu(self):
        updates = dict(
            no_teacher=True,
            freeze_encoder=True,
            gpus=2,
            sortish_sampler=True,
        )
        self._test_distiller_cli(updates, check_contents=False)

    def test_distill_no_teacher(self):
        updates = dict(student_encoder_layers=2, student_decoder_layers=1, no_teacher=True)
        self._test_distiller_cli(updates)

    def test_distill_checkpointing_with_teacher(self):
        updates = dict(
            student_encoder_layers=2,
            student_decoder_layers=1,
            max_epochs=4,
            val_check_interval=0.25,
            alpha_hid=2.0,
            model_name_or_path="IGNORE_THIS_IT_DOESNT_GET_USED",
        )
        model = self._test_distiller_cli(updates, check_contents=False)

        ckpts = list(Path(model.output_dir).glob("*.ckpt"))
        self.assertEqual(1, len(ckpts))
        transformer_ckpts = list(Path(model.output_dir).glob("**/*.bin"))
        self.assertEqual(len(transformer_ckpts), 2)
        examples = lmap(str.strip, model.hparams.data_dir.joinpath("test.source").open().readlines())
        out_path = tempfile.mktemp()
        generate_summaries_or_translations(examples, out_path, str(model.output_dir / "best_tfmr"))
        self.assertTrue(Path(out_path).exists())

        evaluate_checkpoint(ckpts[0], dest_dir=Path(tempfile.mkdtemp()))
        out_path_new = tempfile.mkdtemp()
        convert_pl_to_hf(ckpts[0], transformer_ckpts[0].parent, out_path_new)
        assert os.path.exists(os.path.join(out_path_new, "pytorch_model.bin"))

    def test_loss_fn(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(BART_TINY, return_dict=True)
        input_ids, mask = model.dummy_inputs["input_ids"], model.dummy_inputs["attention_mask"]
        target_ids = torch.tensor([[0, 4, 8, 2], [0, 8, 2, 1]], dtype=torch.long, device=model.device)
        decoder_input_ids = target_ids[:, :-1].contiguous()  # Why this line?
        lm_labels = target_ids[:, 1:].clone()  # why clone?
        model_computed_loss = model(
            input_ids, attention_mask=mask, decoder_input_ids=decoder_input_ids, labels=lm_labels, use_cache=False
        ).loss

        logits = model(input_ids, attention_mask=mask, decoder_input_ids=decoder_input_ids, use_cache=False).logits

        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        smoothed_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, lm_labels, 0.1, ignore_index=model.config.pad_token_id
        )
        with self.assertRaises(AssertionError):
            # TODO: understand why this breaks
            self.assertEqual(nll_loss, model_computed_loss)

    def test_distill_mbart(self):
        updates = dict(
            student_encoder_layers=2,
            student_decoder_layers=1,
            num_train_epochs=4,
            val_check_interval=0.25,
            alpha_hid=2.0,
            task="translation",
            model_name_or_path="IGNORE_THIS_IT_DOESNT_GET_USED",
            tokenizer_name=MBART_TINY,
            teacher=MBART_TINY,
            src_lang="en_XX",
            tgt_lang="ro_RO",
        )
        model = self._test_distiller_cli(updates, check_contents=False)
        assert model.model.config.model_type == "mbart"

        ckpts = list(Path(model.output_dir).glob("*.ckpt"))
        self.assertEqual(1, len(ckpts))
        transformer_ckpts = list(Path(model.output_dir).glob("**/*.bin"))
        all_files = list(Path(model.output_dir).glob("best_tfmr/*"))
        assert len(all_files) > 2
        self.assertEqual(len(transformer_ckpts), 2)

        evaluate_checkpoint(ckpts[0], dest_dir=Path(tempfile.mkdtemp()))

    @unittest.skip("T5 distillation is broken at the moment")
    def test_distill_t5(self):
        updates = dict(
            student_encoder_layers=1,
            student_decoder_layers=1,
            alpha_hid=2.0,
            teacher=T5_TINY,
            model_name_or_path=T5_TINY,
            tokenizer_name=T5_TINY,
        )
        self._test_distiller_cli(updates)

    def _test_distiller_cli(self, updates, check_contents=True):
        default_updates = dict(
            label_smoothing=0.0,
            early_stopping_patience=-1,
            train_batch_size=1,
            eval_batch_size=2,
            max_epochs=2,
            alpha_mlm=0.2,
            alpha_ce=0.8,
            do_predict=True,
            model_name_or_path="sshleifer/tinier_bart",
            teacher=CHEAP_ARGS["model_name_or_path"],
            val_check_interval=0.5,
            alpha_encoder_loss=0.4,
        )
        default_updates.update(updates)
        args_d: dict = CHEAP_ARGS.copy()
        tmp_dir = make_test_data_dir()
        output_dir = tempfile.mkdtemp(prefix="output_")

        args_d.update(data_dir=tmp_dir, output_dir=output_dir, **default_updates)
        model = distill_main(argparse.Namespace(**args_d))
        if not check_contents:
            return model
        contents = os.listdir(output_dir)
        contents = {os.path.basename(p) for p in contents}
        ckpt_files = [p for p in contents if p.endswith("ckpt")]
        assert len(ckpt_files) > 0

        self.assertIn("test_generations.txt", contents)
        self.assertIn("test_results.txt", contents)

        metrics = load_json(model.metrics_save_path)
        last_step_stats = metrics["val"][-1]
        self.assertGreaterEqual(last_step_stats["val_avg_gen_time"], 0.01)
        self.assertGreaterEqual(1.0, last_step_stats["val_avg_gen_time"])
        self.assertIsInstance(last_step_stats[f"val_avg_{model.val_metric}"], float)
        desired_n_evals = int(args_d["max_epochs"] * (1 / args_d["val_check_interval"]) + 1)
        self.assertEqual(len(metrics["val"]), desired_n_evals)
        self.assertEqual(len(metrics["test"]), 1)
        return model


def run_eval_tester(model):
    input_file_name = Path(tempfile.mkdtemp()) / "utest_input.source"
    output_file_name = input_file_name.parent / "utest_output.txt"
    assert not output_file_name.exists()
    articles = [" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County."]
    _dump_articles(input_file_name, articles)
    score_path = str(Path(tempfile.mkdtemp()) / "scores.json")
    task = "translation_en_to_de" if model == T5_TINY else "summarization"
    testargs = f"""
        run_eval_search.py
        {model}
        {input_file_name}
        {output_file_name}
        --score_path {score_path}
        --task {task}
        --num_beams 2
        --length_penalty 2.0
        """.split()

    with patch.object(sys, "argv", testargs):
        run_generate()
        assert Path(output_file_name).exists()
        os.remove(Path(output_file_name))


# test one model to quickly (no-@slow) catch simple problems and do an
# extensive testing of functionality with multiple models as @slow separately
def test_run_eval():
    run_eval_tester(T5_TINY)


# any extra models should go into the list here - can be slow
@slow
@pytest.mark.parametrize("model", [BART_TINY, MBART_TINY])
def test_run_eval_slow(model):
    run_eval_tester(model)


# testing with 2 models to validate: 1. translation (t5) 2. summarization (mbart)
@slow
@pytest.mark.parametrize("model", [T5_TINY, MBART_TINY])
def test_run_eval_search(model):
    input_file_name = Path(tempfile.mkdtemp()) / "utest_input.source"
    output_file_name = input_file_name.parent / "utest_output.txt"
    assert not output_file_name.exists()

    text = {
        "en": ["Machine learning is great, isn't it?", "I like to eat bananas", "Tomorrow is another great day!"],
        "de": [
            "Maschinelles Lernen ist großartig, oder?",
            "Ich esse gerne Bananen",
            "Morgen ist wieder ein toller Tag!",
        ],
    }

    tmp_dir = Path(tempfile.mkdtemp())
    score_path = str(tmp_dir / "scores.json")
    reference_path = str(tmp_dir / "val.target")
    _dump_articles(input_file_name, text["en"])
    _dump_articles(reference_path, text["de"])
    task = "translation_en_to_de" if model == T5_TINY else "summarization"
    testargs = f"""
        run_eval_search.py
        --model_name {model}
        --data_dir {str(input_file_name)}
        --save_dir {str(output_file_name)}
        --score_path {score_path}
        --reference_path {reference_path},
        --task {task}
        --search num_beams=1:2 length_penalty=0.9:1.0
        """.split()

    with patch.object(sys, "argv", testargs):
        with CaptureStdout() as cs:
            run_search()
        expected_strings = [" num_beams | length_penalty", model, "Best score args"]
        un_expected_strings = ["Info"]
        if "translation" in task:
            expected_strings.append("bleu")
        else:
            expected_strings.extend(["rouge1", "rouge2", "rougeL"])
        for w in expected_strings:
            assert w in cs.out
        for w in un_expected_strings:
            assert w not in cs.out
        assert Path(output_file_name).exists()
        os.remove(Path(output_file_name))


@pytest.mark.parametrize(
    "model",
    [T5_TINY, BART_TINY, MBART_TINY, MARIAN_TINY, FSMT_TINY],
)
def test_finetune(model):
    args_d: dict = CHEAP_ARGS.copy()
    task = "translation" if model in [MBART_TINY, MARIAN_TINY, FSMT_TINY] else "summarization"
    args_d["label_smoothing"] = 0.1 if task == "translation" else 0

    tmp_dir = make_test_data_dir()
    output_dir = tempfile.mkdtemp(prefix="output_")
    args_d.update(
        data_dir=tmp_dir,
        model_name_or_path=model,
        tokenizer_name=None,
        train_batch_size=2,
        eval_batch_size=2,
        output_dir=output_dir,
        do_predict=True,
        task=task,
        src_lang="en_XX",
        tgt_lang="ro_RO",
        freeze_encoder=True,
        freeze_embeds=True,
    )
    assert "n_train" in args_d
    args = argparse.Namespace(**args_d)
    module = main(args)

    input_embeds = module.model.get_input_embeddings()
    assert not input_embeds.weight.requires_grad
    if model == T5_TINY:
        lm_head = module.model.lm_head
        assert not lm_head.weight.requires_grad
        assert (lm_head.weight == input_embeds.weight).all().item()
    elif model == FSMT_TINY:
        fsmt = module.model.model
        embed_pos = fsmt.decoder.embed_positions
        # XXX: about to change to embed_pos.weight.requires_grad
        assert not embed_pos.weights.requires_grad
        assert not fsmt.decoder.embed_tokens.weight.requires_grad
        # check that embeds are not the same
        assert fsmt.decoder.embed_tokens != fsmt.encoder.embed_tokens
    else:
        bart = module.model.model
        embed_pos = bart.decoder.embed_positions
        assert not embed_pos.weight.requires_grad
        assert not bart.shared.weight.requires_grad
        # check that embeds are the same
        assert bart.decoder.embed_tokens == bart.encoder.embed_tokens
        assert bart.decoder.embed_tokens == bart.shared


def test_finetune_extra_model_args():
    args_d: dict = CHEAP_ARGS.copy()

    task = "summarization"
    tmp_dir = make_test_data_dir()

    args_d.update(
        data_dir=tmp_dir,
        tokenizer_name=None,
        train_batch_size=2,
        eval_batch_size=2,
        do_predict=False,
        task=task,
        src_lang="en_XX",
        tgt_lang="ro_RO",
        freeze_encoder=True,
        freeze_embeds=True,
    )

    # test models whose config includes the extra_model_args
    model = BART_TINY
    output_dir = tempfile.mkdtemp(prefix="output_1_")
    args_d1 = args_d.copy()
    args_d1.update(
        model_name_or_path=model,
        output_dir=output_dir,
    )
    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        args_d1[p] = 0.5
    args = argparse.Namespace(**args_d1)
    model = main(args)
    for p in extra_model_params:
        assert getattr(model.config, p) == 0.5, f"failed to override the model config for param {p}"

    # test models whose config doesn't include the extra_model_args
    model = T5_TINY
    output_dir = tempfile.mkdtemp(prefix="output_2_")
    args_d2 = args_d.copy()
    args_d2.update(
        model_name_or_path=model,
        output_dir=output_dir,
    )
    unsupported_param = "encoder_layerdrop"
    args_d2[unsupported_param] = 0.5
    args = argparse.Namespace(**args_d2)
    with pytest.raises(Exception) as excinfo:
        model = main(args)
    assert str(excinfo.value) == f"model config doesn't have a `{unsupported_param}` attribute"


def test_finetune_lr_schedulers():
    args_d: dict = CHEAP_ARGS.copy()

    task = "summarization"
    tmp_dir = make_test_data_dir()

    model = BART_TINY
    output_dir = tempfile.mkdtemp(prefix="output_1_")

    args_d.update(
        data_dir=tmp_dir,
        model_name_or_path=model,
        output_dir=output_dir,
        tokenizer_name=None,
        train_batch_size=2,
        eval_batch_size=2,
        do_predict=False,
        task=task,
        src_lang="en_XX",
        tgt_lang="ro_RO",
        freeze_encoder=True,
        freeze_embeds=True,
    )

    # emulate finetune.py
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())
    args = {"--help": True}

    # --help test
    with pytest.raises(SystemExit) as excinfo:
        with CaptureStdout() as cs:
            args = parser.parse_args(args)
        assert False, "--help is expected to sys.exit"
    assert excinfo.type == SystemExit
    expected = lightning_base.arg_to_scheduler_metavar
    assert expected in cs.out, "--help is expected to list the supported schedulers"

    # --lr_scheduler=non_existing_scheduler test
    unsupported_param = "non_existing_scheduler"
    args = {f"--lr_scheduler={unsupported_param}"}
    with pytest.raises(SystemExit) as excinfo:
        with CaptureStderr() as cs:
            args = parser.parse_args(args)
        assert False, "invalid argument is expected to sys.exit"
    assert excinfo.type == SystemExit
    expected = f"invalid choice: '{unsupported_param}'"
    assert expected in cs.err, f"should have bailed on invalid choice of scheduler {unsupported_param}"

    # --lr_scheduler=existing_scheduler test
    supported_param = "cosine"
    args_d1 = args_d.copy()
    args_d1["lr_scheduler"] = supported_param
    args = argparse.Namespace(**args_d1)
    model = main(args)
    assert getattr(model.hparams, "lr_scheduler") == supported_param, f"lr_scheduler={supported_param} shouldn't fail"
