# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" TF 2.0 BERT model. """


from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from .activations_tf import get_tf_activation
from .configuration_bert import BertConfig
from .file_utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from .modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFCausalLMOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from .modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMaskedLanguageModelingLoss,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    input_analysis,
    keras_serializable,
    shape_list,
)
from .utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


class TFBertPreTrainingLoss:
    """
    Loss function suitable for BERT-like pre-training, that is, the task of pretraining a language model by combining NSP + MLM.

    .. note::

         Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.
    """

    def compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to -100
        # are taken into account as loss
        masked_lm_active_loss = tf.not_equal(x=tf.reshape(tensor=labels["labels"], shape=(-1,)), y=-100)
        masked_lm_reduced_logits = tf.boolean_mask(
            tensor=tf.reshape(tensor=logits[0], shape=(-1, shape_list(tensor=logits[0])[2])),
            mask=masked_lm_active_loss,
        )
        masked_lm_labels = tf.boolean_mask(
            tensor=tf.reshape(tensor=labels["labels"], shape=(-1,)), mask=masked_lm_active_loss
        )
        next_sentence_active_loss = tf.not_equal(
            x=tf.reshape(tensor=labels["next_sentence_label"], shape=(-1,)), y=-100
        )
        next_sentence_reduced_logits = tf.boolean_mask(
            tensor=tf.reshape(tensor=logits[1], shape=(-1, 2)), mask=next_sentence_active_loss
        )
        next_sentence_label = tf.boolean_mask(
            tensor=tf.reshape(tensor=labels["next_sentence_label"], shape=(-1,)), mask=next_sentence_active_loss
        )
        masked_lm_loss = loss_fn(y_true=masked_lm_labels, y_pred=masked_lm_reduced_logits)
        next_sentence_loss = loss_fn(y_true=next_sentence_label, y_pred=next_sentence_reduced_logits)
        masked_lm_loss = tf.reshape(tensor=masked_lm_loss, shape=(-1, shape_list(tensor=next_sentence_loss)[0]))
        masked_lm_loss = tf.reduce_mean(input_tensor=masked_lm_loss, axis=0)

        return masked_lm_loss + next_sentence_loss


class WordEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._initializer_range = config.initializer_range

    def build(self, input_shape):
        self.word_embeddings = self.add_weight(
            name="weight",
            shape=[self._vocab_size, self._hidden_size],
            initializer=get_initializer(initializer_range=self._initializer_range),
            dtype=tf.float32,
        )

        super().build(input_shape=input_shape)

    def call(self, input_ids):
        flat_input_ids = tf.reshape(tensor=input_ids, shape=[-1])
        embeddings = tf.gather(params=self.word_embeddings, indices=flat_input_ids)
        embeddings = tf.reshape(tensor=embeddings, shape=tf.concat([tf.shape(input_ids), [self._hidden_size]], axis=0))

        embeddings.set_shape(shape=input_ids.shape.as_list() + [self._hidden_size])

        return embeddings


class TokenTypeEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self._type_vocab_size = config.type_vocab_size
        self._hidden_size = config.hidden_size
        self._initializer_range = config.initializer_range

    def build(self, input_shape):
        self.token_type_embeddings = self.add_weight(
            name="embeddings",
            shape=[self._type_vocab_size, self._hidden_size],
            initializer=get_initializer(initializer_range=self._initializer_range),
            dtype=tf.float32,
        )

        super().build(input_shape=input_shape)

    def call(self, token_type_ids):
        flat_token_type_ids = tf.reshape(tensor=token_type_ids, shape=[-1])
        one_hot_data = tf.one_hot(indices=flat_token_type_ids, depth=self._type_vocab_size, dtype=self._compute_dtype)
        embeddings = tf.matmul(a=one_hot_data, b=self.token_type_embeddings)
        embeddings = tf.reshape(
            tensor=embeddings, shape=tf.concat([tf.shape(token_type_ids), [self._hidden_size]], axis=0)
        )

        embeddings.set_shape(shape=token_type_ids.shape.as_list() + [self._hidden_size])

        return embeddings


class PositionEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self._max_position_embeddings = config.max_position_embeddings
        self._hidden_size = config.hidden_size
        self._initializer_range = config.initializer_range

    def build(self, input_shape):
        self._position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self._max_position_embeddings, self._hidden_size],
            initializer=get_initializer(initializer_range=self._initializer_range),
        )

        super().build(input_shape)

    def call(self, position_ids):
        input_shape = tf.shape(input=position_ids)
        position_embeddings = self._position_embeddings[: input_shape[1], :]

        return tf.broadcast_to(input=position_embeddings, shape=input_shape)


class TFBertEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self.word_embeddings = WordEmbeddings(config, name="word_embeddings")
        self.position_embeddings = PositionEmbeddings(config, name="position_embeddings")
        self.token_type_embeddings = TokenTypeEmbeddings(config, name="token_type_embeddings")
        self.embeddings = tf.keras.layers.Add()
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        mode="embedding",
        training=False,
    ):
        """
        Get token embeddings of inputs.

        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".

        Returns:
            outputs: If mode == "embedding", output embedding tensor, float32 with shape [batch_size, length,
            embedding_size]; if mode == "linear", output linear tensor, float32 with shape [batch_size, length,
            vocab_size].

        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == "embedding":
            return self._embedding(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                training=training,
            )
        elif mode == "linear":
            return self._linear(input_ids=input_ids)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, input_ids, position_ids, token_type_ids, inputs_embeds, training=False):
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is None:
            tmp_word_embeddings = inputs_embeds
        else:
            tmp_word_embeddings = self.word_embeddings(input_ids=input_ids)

        if token_type_ids is None:
            if input_ids is not None:
                input_shape = shape_list(input_ids)
            else:
                input_shape = shape_list(inputs_embeds)[:-1]

            token_type_ids = tf.fill(dims=input_shape, value=0)

        tmp_position_embeddings = self.position_embeddings(position_ids=tmp_word_embeddings)
        tmp_token_type_embeddings = self.token_type_embeddings(token_type_ids)
        final_embeddings = self.embeddings(
            inputs=[tmp_word_embeddings, tmp_position_embeddings, tmp_token_type_embeddings]
        )
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(final_embeddings, training=training)

    def _linear(self, inputs):
        """
        Computes logits by running inputs through a linear layer.

        Args:
            inputs: A float32 tensor with shape [batch_size, length, hidden_size].

        Returns:
            float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = shape_list(tensor=input_ids)[0]
        length = shape_list(tensor=input_ids)[1]
        x = tf.reshape(tensor=input_ids, shape=(-1, self._hidden_size))
        logits = tf.matmul(a=x, b=self.word_embeddings.word_embeddings, transpose_b=True)

        return tf.reshape(tensor=logits, shape=(batch_size, length, self._vocab_size))


class TFBertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, config.num_attention_heads, self.attention_head_size),
            bias_axes="de",
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="query",
        )
        self.key = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, config.num_attention_heads, self.attention_head_size),
            bias_axes="de",
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, config.num_attention_heads, self.attention_head_size),
            bias_axes="de",
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="value",
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

    def call(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, training=False):
        query_tensor = self.query(inputs=hidden_states)

        # `key_tensor` = [B, S, N, H]
        key_tensor = self.key(inputs=hidden_states)

        # `value_tensor` = [B, S, N, H]
        value_tensor = self.value(inputs=hidden_states)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum("bsnh,btnh->bnts", key_tensor, query_tensor)
        dk = tf.cast(x=self.attention_head_size, dtype=attention_scores.dtype)  # scale attention_scores
        attention_scores = tf.multiply(x=attention_scores, y=tf.math.rsqrt(x=dk))

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        attention_output = tf.einsum("bnts,bsnh->btnh", attention_probs, value_tensor)
        outputs = (attention_output, attention_probs)

        return outputs


class TFBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dense = tf.keras.layers.experimental.EinsumDense(
            equation="abcd,cde->abe",
            output_shape=(None, self.all_head_size),
            bias_axes="e",
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="dense",
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class TFBertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TFBertSelfAttention(config=config, name="self")
        self.dense_output = TFBertSelfOutput(config=config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, training=False):
        self_outputs = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0],
            input_tensor=hidden_states,
            training=training,
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


class TFBertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cd->abd",
            output_shape=(None, config.intermediate_size),
            bias_axes="d",
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="dense",
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(activation_string=config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(inputs=hidden_states)

        return hidden_states


class TFBertOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.experimental.EinsumDense(
            equation="abc,cd->abd",
            bias_axes="d",
            output_shape=(None, config.hidden_size),
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class TFBertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFBertAttention(config=config, name="attention")
        self.intermediate = TFBertIntermediate(config=config, name="intermediate")
        self.bert_output = TFBertOutput(config=config, name="output")

    def call(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, training=False):
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output,
            input_tensor=attention_output,
            training=training,
        )
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs


class TFBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.layer = [TFBertLayer(config=config, name="layer_._{}".format(i)) for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        training=False,
    ):
        all_hidden_states = tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True, element_shape=hidden_states.shape
        )
        all_attentions = None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states.write(index=i, value=hidden_states)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if all_attentions is None:
                all_attentions = tf.TensorArray(
                    dtype=tf.float32, size=0, dynamic_size=True, element_shape=layer_outputs[1].shape
                )

            if output_attentions:
                all_attentions = all_attentions.write(index=i, value=layer_outputs[1])

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states.write(index=all_hidden_states.size(), value=hidden_states)

        all_attentions = all_attentions.stack()
        all_hidden_states = all_hidden_states.stack()

        if tf.executing_eagerly():
            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class TFBertPooler(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            activation="tanh",
            name="dense",
        )

    def call(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output


class TFBertPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(activation_string=config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

    def call(self, hidden_states):
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.transform_act_fn(inputs=hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states


class TFBertLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.transform = TFBertPredictionHeadTransform(config=config, name="transform")

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")

        super().build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states=hidden_states)
        hidden_states = self.input_embeddings(input_ids=hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias

        return hidden_states


class TFBertOnlyMLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.predictions = TFBertLMPredictionHead(config=config, input_embeddings=input_embeddings, name="predictions")

    def call(self, sequence_output):
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores


class TFBertOnlyNSPHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.seq_relationship = tf.keras.layers.Dense(
            units=2,
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="seq_relationship",
        )

    def call(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)

        return seq_relationship_score


class TFBertPreTrainingHeads(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.predictions = TFBertLMPredictionHead(config=config, input_embeddings=input_embeddings, name="predictions")
        self.seq_relationship = tf.keras.layers.Dense(
            units=2,
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="seq_relationship",
        )

    def call(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(hidden_states=sequence_output)
        seq_relationship_score = self.seq_relationship(inputs=pooled_output)

        return prediction_scores, seq_relationship_score


@keras_serializable
class TFBertMainLayer(tf.keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.embeddings = TFBertEmbeddings(config=config, name="embeddings")
        self.encoder = TFBertEncoder(config=config, name="encoder")
        self.pooler = TFBertPooler(config=config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings.word_embeddings = value
        self.embeddings._vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        training=False,
    ):

        inputs = input_analysis(
            func=self.call,
            inputs=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        output_attentions = (
            inputs["output_attentions"] if inputs["output_attentions"] is not None else self.output_attentions
        )
        output_hidden_states = (
            inputs["output_hidden_states"] if inputs["output_hidden_states"] is not None else self.output_hidden_states
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.return_dict

        if not return_dict and not tf.executing_eagerly():
            tf.print(
                "Since 🤗 Transformers v4.0, `return_dict=False` is deprecated and cannot be used in graph mode. Setting `return_dict=True`. Use TensorFlow eager mode to enable the `return_dict=False` behavior."
            )
            return_dict = True

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(tensor=inputs["input_ids"])
        elif inputs_embeds is not None:
            input_shape = shape_list(tensor=inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(dims=input_shape, value=1)

        if inputs["token_type_ids"] is None:
            inputs["token_type_ids"] = tf.fill(dims=input_shape, value=0)

        embedding_output = self.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            inputs_embeds=inputs["inputs_embeds"],
            training=inputs["training"],
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = inputs["attention_mask"][:, tf.newaxis, tf.newaxis, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(x=extended_attention_mask, dtype=embedding_output.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=inputs["head_mask"],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=inputs["training"],
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None

        if tf.executing_eagerly():
            if not return_dict:
                return (
                    sequence_output,
                    pooled_output,
                ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"


@dataclass
class TFBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFBertForPreTraining`.

    Args:
        prediction_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[tf.Tensor] = None
    prediction_logits: tf.Tensor = None
    seq_relationship_logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having all
        the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
        the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Args:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.TFPreTrainedModel.from_pretrained` method to load the
            model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`tf.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. Since
            the v4.0 this parameter is always set to :obj:`True` in graph mode. The :obj:`False` value is deprecated and
            this parameter will be removed in a future major version.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class TFBertModel(TFBertPreTrainedModel):
    authorized_unexpected_keys = [r"cls"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = TFBertMainLayer(config=config, name="bert")

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-cased",
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        training=False,
    ):
        inputs = input_analysis(
            func=self.call,
            inputs=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.bert.return_dict
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )

        return outputs


@add_start_docstrings(
    """
Bert Model with two heads on top as done during the pre-training:
    a `masked language modeling` head and a `next sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForPreTraining(TFBertPreTrainedModel, TFBertPreTrainingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.initializer_range = config.initializer_range
        self.bert = TFBertMainLayer(config=config, name="bert")
        self.cls = TFBertPreTrainingHeads(config=config, input_embeddings=self.bert.embeddings, name="cls")

    def get_output_embeddings(self):
        return self.bert.embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        labels=None,
        next_sentence_label=None,
        training=False,
    ):
        r"""
        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import BertTokenizer, TFBertForPreTraining

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = TFBertForPreTraining.from_pretrained('bert-base-uncased')
            >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
            >>> outputs = model(input_ids)
            >>> prediction_scores, seq_relationship_scores = outputs[:2]

        """

        inputs = input_analysis(
            func=self.call,
            inputs=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            next_sentence_label=next_sentence_label,
            training=training,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.bert.return_dict
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output=sequence_output, pooled_output=pooled_output
        )
        total_loss = None

        if inputs["labels"] is not None and inputs["next_sentence_label"] is not None:
            d_labels = {"labels": inputs["labels"]}
            d_labels["next_sentence_label"] = inputs["next_sentence_label"]
            total_loss = self.compute_loss(labels=d_labels, logits=(prediction_scores, seq_relationship_score))

        if tf.executing_eagerly():
            if not return_dict:
                return (prediction_scores, seq_relationship_score) + outputs[2:]

        return TFBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a `language modeling` head on top for CLM fine-tuning. """, BERT_START_DOCSTRING
)
class TFBertLMHeadModel(TFBertPreTrainedModel, TFCausalLanguageModelingLoss):
    authorized_unexpected_keys = [r"pooler", r"seq_relationship"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if not config.is_decoder:
            logger.warning("If you want to use `TFBertLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.initializer_range = config.initializer_range
        self.bert = TFBertMainLayer(config=config, add_pooling_layer=False, name="bert")
        self.cls = TFBertOnlyMLMHead(config=config, input_embeddings=self.bert.embeddings, name="cls")

    def get_output_embeddings(self):
        return self.bert.embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-cased",
        output_type=TFCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        labels=None,
        training=False,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        inputs = input_analysis(
            func=self.call,
            inputs=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.bert.return_dict
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output=sequence_output, training=training)
        lm_loss = None

        if inputs["labels"] is not None:
            # shift labels to the left and cut last logit token
            shifted_prediction_scores = prediction_scores[:, :-1]
            labels = inputs["labels"][:, 1:]
            lm_loss = self.compute_loss(labels=labels, logits=shifted_prediction_scores)

        if tf.executing_eagerly():
            if not return_dict:
                output = (prediction_scores,) + outputs[2:]

                return ((lm_loss,) + output) if lm_loss is not None else output

        return TFCausalLMOutput(
            loss=lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class TFBertForMaskedLM(TFBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    authorized_unexpected_keys = [r"pooler", r"seq_relationship"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if config.is_decoder:
            logger.warning(
                "If you want to use `TFBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.initializer_range = config.initializer_range
        self.bert = TFBertMainLayer(config=config, add_pooling_layer=False, name="bert")
        self.cls = TFBertOnlyMLMHead(config=config, input_embeddings=self.bert.embeddings, name="cls")

    def get_output_embeddings(self):
        return self.bert.embeddings

    def resize_token_embeddings(self, new_num_tokens):
        super().resize_token_embeddings(new_num_tokens=new_num_tokens)

        if new_num_tokens is not None:
            self.cls.predictions.bias = self.add_weight(
                shape=(new_num_tokens,), initializer="zeros", trainable=True, name="bias"
            )

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-cased",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        labels=None,
        training=False,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """

        inputs = input_analysis(
            func=self.call,
            inputs=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.bert.return_dict
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output=sequence_output, training=training)
        masked_lm_loss = (
            None if inputs["labels"] is None else self.compute_loss(labels=inputs["labels"], logits=prediction_scores)
        )

        if tf.executing_eagerly():
            if not return_dict:
                output = (prediction_scores,) + outputs[2:]

                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return TFMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top. """,
    BERT_START_DOCSTRING,
)
class TFBertForNextSentencePrediction(TFBertPreTrainedModel, TFNextSentencePredictionLoss):
    authorized_unexpected_keys = [r"predictions"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = TFBertMainLayer(config, name="bert")
        self.nsp = TFBertNSPHead(config, name="nsp___cls")

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        next_sentence_label=None,
        training=False,
    ):
        r"""
        Return:
        Examples::
            >>> import tensorflow as tf
            >>> from transformers import BertTokenizer, TFBertForNextSentencePrediction
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased')
            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='tf')
            >>> logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]
            >>> assert logits[0][0] < logits[0][1] # the next sentence was random
        """
        return_dict = return_dict if return_dict is not None else self.bert.return_dict

        inputs = input_analysis(
            func=self.call,
            inputs=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            next_sentence_label=next_sentence_label,
            training=training,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.bert.return_dict
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )
        pooled_output = outputs[1]
        seq_relationship_scores = self.nsp(pooled_output)

        next_sentence_loss = (
            None
            if inputs["next_sentence_label"] is None
            else self.compute_loss(labels=inputs["next_sentence_label"], logits=seq_relationship_scores)
        )

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return TFNextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForSequenceClassification(TFBertPreTrainedModel, TFSequenceClassificationLoss):
    authorized_unexpected_keys = [r"cls"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.bert = TFBertMainLayer(config=config, name="bert")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="classifier",
        )

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-cased",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        labels=None,
        training=False,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        inputs = input_analysis(
            func=self.call,
            inputs=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.bert.return_dict
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=inputs["training"])
        logits = self.classifier(inputs=pooled_output)
        loss = None if inputs["labels"] is None else self.compute_loss(labels=inputs["labels"], logits=logits)

        if tf.executing_eagerly():
            if not return_dict:
                output = (logits,) + outputs[2:]

                return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForMultipleChoice(TFBertPreTrainedModel, TFMultipleChoiceLoss):
    authorized_unexpected_keys = [r"cls"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = TFBertMainLayer(config=config, name="bert")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="classifier",
        )

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        return {"input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS)}

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-cased",
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        labels=None,
        training=False,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """

        inputs = input_analysis(
            func=self.call,
            inputs=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
        )

        if inputs["input_ids"] is not None:
            num_choices = shape_list(tensor=inputs["input_ids"])[1]
            seq_length = shape_list(tensor=inputs["input_ids"])[2]
        else:
            num_choices = shape_list(tensor=inputs["inputs_embeds"])[1]
            seq_length = shape_list(tensor=inputs["inputs_embeds"])[2]

        flat_input_ids = (
            tf.reshape(tensor=inputs["input_ids"], shape=(-1, seq_length)) if inputs["input_ids"] is not None else None
        )
        flat_attention_mask = (
            tf.reshape(tensor=inputs["attention_mask"], shape=(-1, seq_length))
            if inputs["attention_mask"] is not None
            else None
        )
        flat_token_type_ids = (
            tf.reshape(tensor=inputs["token_type_ids"], shape=(-1, seq_length))
            if inputs["token_type_ids"] is not None
            else None
        )
        flat_position_ids = (
            tf.reshape(tensor=inputs["position_ids"], shape=(-1, seq_length))
            if inputs["position_ids"] is not None
            else None
        )
        flat_inputs_embeds = (
            tf.reshape(
                tensor=inputs["inputs_embeds"], shape=(-1, seq_length, shape_list(tensor=inputs["inputs_embeds"])[3])
            )
            if inputs["inputs_embeds"] is not None
            else None
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.bert.return_dict
        outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            head_mask=inputs["head_mask"],
            inputs_embeds=flat_inputs_embeds,
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=inputs["training"])
        logits = self.classifier(inputs=pooled_output)
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))
        loss = None if inputs["labels"] is None else self.compute_loss(labels=inputs["labels"], logits=reshaped_logits)

        if tf.executing_eagerly():
            if not return_dict:
                output = (reshaped_logits,) + outputs[2:]

                return ((loss,) + output) if loss is not None else output

        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForTokenClassification(TFBertPreTrainedModel, TFTokenClassificationLoss):
    authorized_unexpected_keys = [r"pooler", r"cls"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.bert = TFBertMainLayer(config=config, add_pooling_layer=False, name="bert")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="classifier",
        )

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-cased",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        labels=None,
        training=False,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """

        inputs = input_analysis(
            func=self.call,
            inputs=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.bert.return_dict
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(inputs=sequence_output, training=inputs["training"])
        logits = self.classifier(inputs=sequence_output)
        loss = None if inputs["labels"] is None else self.compute_loss(labels=inputs["labels"], logits=logits)

        if tf.executing_eagerly():
            if not return_dict:
                output = (logits,) + outputs[2:]

                return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
class TFBertForQuestionAnswering(TFBertPreTrainedModel, TFQuestionAnsweringLoss):
    authorized_unexpected_keys = [r"pooler", r"cls"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.bert = TFBertMainLayer(config=config, add_pooling_layer=False, name="bert")
        self.qa_outputs = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(initializer_range=config.initializer_range),
            name="qa_outputs",
        )

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-cased",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        start_positions=None,
        end_positions=None,
        training=False,
    ):
        r"""
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """

        inputs = input_analysis(
            func=self.call,
            inputs=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start_positions=start_positions,
            end_positions=end_positions,
            training=training,
        )
        return_dict = inputs["return_dict"] if inputs["return_dict"] is not None else self.bert.return_dict
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=return_dict,
            training=inputs["training"],
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(inputs=sequence_output)
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        total_loss = None

        if inputs["start_positions"] is not None and inputs["end_positions"] is not None:
            labels = {"start_position": inputs["start_positions"]}
            labels["end_position"] = inputs["end_positions"]
            total_loss = self.compute_loss(labels=labels, logits=(start_logits, end_logits))

        if tf.executing_eagerly():
            if not return_dict:
                output = (start_logits, end_logits) + outputs[2:]

                return ((total_loss,) + output) if total_loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
