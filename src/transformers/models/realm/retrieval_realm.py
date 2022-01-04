# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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
"""Realm Retriever model implementation."""
import numpy as np

from ...utils import logging


logger = logging.get_logger(__name__)


def convert_tfrecord_to_np(block_records_path, num_block_records):
    import tensorflow.compat.v1 as tf

    blocks_dataset = tf.data.TFRecordDataset(block_records_path, buffer_size=512 * 1024 * 1024)
    blocks_dataset = blocks_dataset.batch(num_block_records, drop_remainder=True)
    np_record = next(blocks_dataset.take(1).as_numpy_iterator())

    return np_record


class ScaNNSearcher:
    def __init__(
        self,
        db,
        num_neighbors,
        dimensions_per_block=2,
        num_leaves=1000,
        num_leaves_to_search=100,
        training_sample_size=100000,
    ):
        """Build scann searcher."""

        from scann.scann_ops.py.scann_ops_pybind import builder as Builder

        builder = Builder(db=db, num_neighbors=num_neighbors, distance_measure="dot_product")
        builder = builder.tree(
            num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=training_sample_size
        )
        builder = builder.score_ah(dimensions_per_block=dimensions_per_block)

        self.searcher = builder.build()

    def search_batched(self, question_projection):
        retrieved_block_ids, _ = self.searcher.search_batched(question_projection.detach().cpu())
        return retrieved_block_ids.astype("int64")


class RealmRetriever:
    """The retriever of REALM outputting the retrieved evidence block and whether the block has answers."

    Parameters:
        config ([`RealmConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        tokenizer ([`RealmTokenizer`]): The tokenizer to encode retrieved texts.
        block_records_path (`str`): The path of `block_records`, which cantains evidence texts.
    """

    def __init__(self, config, tokenizer, block_records_path):
        super().__init__()
        self.config = config
        self.block_records = convert_tfrecord_to_np(
            block_records_path=block_records_path,
            num_block_records=config.num_block_records,
        )
        self.tokenizer = tokenizer

    def __call__(self, retrieved_block_ids, question_input_ids, answer_ids, return_tensors="pt"):
        retrieved_blocks = np.take(self.block_records, indices=retrieved_block_ids, axis=0)

        question = self.tokenizer.decode(question_input_ids[0], skip_special_tokens=True)

        text = []
        text_pair = []
        for retrieved_block in retrieved_blocks:
            text.append(question)
            text_pair.append(retrieved_block.decode())

        concat_inputs = self.tokenizer(
            text, text_pair, padding=True, truncation=True, max_length=self.config.reader_seq_len
        )
        concat_inputs_tensors = concat_inputs.convert_to_tensors(return_tensors)

        if answer_ids is not None:
            return self.block_has_answer(concat_inputs, answer_ids) + (concat_inputs_tensors,)
        else:
            return (None, None, None, concat_inputs_tensors)

    def block_has_answer(self, concat_inputs, answer_ids):
        """check if retrieved_blocks has answers."""
        has_answers = []
        start_pos = []
        end_pos = []
        max_answers = 0

        for input_id in concat_inputs.input_ids:
            start_pos.append([])
            end_pos.append([])
            input_id_list = input_id.tolist()
            # Checking answers after the [SEP] token
            sep_idx = input_id_list.index(self.tokenizer.sep_token_id)
            for answer in answer_ids:
                for idx in range(sep_idx, len(input_id)):
                    if answer[0] == input_id_list[idx]:
                        if input_id_list[idx : idx + len(answer)] == answer:
                            start_pos[-1].append(idx)
                            end_pos[-1].append(idx + len(answer) - 1)

            if len(start_pos[-1]) == 0:
                has_answers.append(False)
            else:
                has_answers.append(True)
                if len(start_pos[-1]) > max_answers:
                    max_answers = len(start_pos[-1])

        # Pad -1 to max_answers
        for start_pos_, end_pos_ in zip(start_pos, end_pos):
            if len(start_pos_) < max_answers:
                padded = [-1] * (max_answers - len(start_pos_))
                start_pos_ += padded
                end_pos_ += padded

        return has_answers, start_pos, end_pos
