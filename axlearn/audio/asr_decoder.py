# Copyright © 2023 Apple Inc.

"""ASR decoder layers."""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from axlearn.common import struct
from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.config import (
    REQUIRED,
    ConfigOr,
    InstantiableConfig,
    Required,
    config_class,
    maybe_instantiate,
)
from axlearn.common.decoding import (
    NEG_INF,
    PrefixMerger,
    StopOnSubsequence,
    add_decoding_dim,
    beam_search_decode,
    compute_merge_matrix_by_prefix_ids,
    flatten_decoding_dim,
    infer_initial_time_step,
    sample_decode,
    unflatten_decoding_dim,
)
from axlearn.common.layers import Embedding, Linear
from axlearn.common.logit_modifiers import LogitsToLogitsFn
from axlearn.common.module import Module, child_context
from axlearn.common.rnn import BaseRNNCell, LSTMCell
from axlearn.common.transducer import Transducer, log_probs_from_blank_and_tokens
from axlearn.common.utils import Nested, NestedTensor, Tensor, vectorized_tree_map


def _is_valid_ctc_seq(
    *, paddings: Tensor, target_labels: Tensor, target_paddings: Tensor
) -> Tensor:
    """Returns whether each input sequence passes validity check.

    Note that `optax.ctc_loss` returns -logeps (default to 1e5) if the
    input length is smaller than the label length plus number of
    consecutive duplications, because we need a blank label to transition
    between the same labels. When this condition is not met, it should be
    considered as an invalid sequence and the loss should be ignored.

    A validity check is passed if for an example when:
        input.length >= labels.length + num(consecutive dup label tokens)

    Args:
        paddings: A 0/1 tensor of shape [batch_size, num_frames], indicating whether
            an input frame is a padding.
        target_labels: A int32 tensor of shape [batch_size, num_frames].
        target_paddings: A 0/1 tensor of shape [batch_size, num_frames], indicating
            whether a label is a padding. Note that at the moment, `target_paddings`
            must be left-justified, i.e., it must starts with 0 and followed by 1, and
            not transition back to 0.
            TODO(yqw): support generic target_paddings.

    Returns:
        A float tensor of [batch_size, ] indicating if each (input, label) pair is valid,
        with a value of 1.0 indicating valid and 0.0 otherwise.
    """
    # [batch_size, ]
    label_lengths = jnp.sum(1.0 - target_paddings, axis=-1)
    # [batch_size, ]
    input_lengths = jnp.sum(1.0 - paddings, axis=-1)
    # [batch_size, num_frames - 1]
    dups = (1.0 - target_paddings[:, 1:]) * (target_labels[:, :-1] == target_labels[:, 1:])
    # [batch_size, ]
    num_consecutive_dups = jnp.sum(dups, axis=-1)
    # [batch_size, ]
    is_valid = (label_lengths + num_consecutive_dups) <= input_lengths
    return is_valid


class CTCPrefixMerger(PrefixMerger):
    """Merges equivalent lower-ranked beams into higher-ranked ones.

    Beams are compared after removing repeats and blanks following CTC.

    See Section 3.1 of https://dl.acm.org/doi/10.1145/1143844.1143891.
    """

    def __init__(self, blank_id: int):
        self._blank_id = blank_id

    def init_state(self, *, tokens: Tensor) -> Nested[Tensor]:
        """Initializes the prefix merger state from the initial prefix `tokens`.

        If the initial prefix is non-empty, we produce state equivalent to initializing from an
        empty prefix and invoking `update` token-by-token until the end of the initial prefix.
        """
        outputs = _map_label_sequences(tokens, blank_id=self._blank_id, pad_id=-1)
        # Compute last tokens.
        last_token = jnp.take_along_axis(outputs["sequences"], outputs["lengths"] - 1, axis=-1)
        return dict(
            sequences=outputs["sequences"],
            last_token=jnp.squeeze(last_token, axis=2),
            lengths=jnp.squeeze(outputs["lengths"], axis=2),
        )

    def compute(self, state: Nested[Tensor]) -> Tensor:
        """Computes a merge matrix by comparing prefixes."""
        return compute_merge_matrix_by_prefix_ids(state["sequences"])

    def update(self, *, tokens: Tensor, state: Nested[Tensor]) -> Nested[Tensor]:
        """Updates prefix merger state given the next candidate token."""

        def _update_seq(token: Tensor, seq_state: Nested[Tensor]) -> Nested[Tensor]:
            skip = jnp.logical_or(token == seq_state["last_token"], token == self._blank_id)
            return dict(
                sequences=jnp.where(
                    skip,
                    seq_state["sequences"],
                    jax.lax.dynamic_update_index_in_dim(
                        seq_state["sequences"], token, seq_state["lengths"], axis=0
                    ),
                ),
                last_token=token,
                lengths=seq_state["lengths"] + jnp.where(skip, 0, 1),
            )

        # vmap over both `batch_size` and `num_decodes`.
        return jax.vmap(jax.vmap(_update_seq))(tokens, state)


class DecodeOutputs(struct.PyTreeNode):
    """Output of decoding."""

    # Raw decode output sequences. May contain blank and/or repeated tokens.
    # An int Tensor of shape [batch_size, num_decodes, max_decode_len].
    raw_sequences: Tensor
    # Post-processed sequences, e.g. after removing blanks or repeated tokens.
    # An int Tensor of shape [batch_size, num_decodes, max_decode_len].
    sequences: Tensor
    # Paddings of the post-processed sequences.
    # A 0/1 Tensor of shape [batch_size, num_decodes, max_decode_len].
    paddings: Tensor
    # Scores corresponding to sequences above (log probabilities).
    # A float Tensor of shape [batch_size, num_decodes].
    scores: Tensor


class ASRDecoderModelBase(BaseModel):
    """ASR decoder model base."""

    @config_class
    class Config(BaseModel.Config):
        """Configures CTCDecoderModel."""

        # Dimensionality of inputs.
        dim: Required[int] = REQUIRED
        # The vocab size.
        vocab_size: Required[int] = REQUIRED
        # Blank token ID.
        blank_token_id: int = 0

    def forward(
        self,
        input_batch: Nested[Tensor],
    ) -> Tuple[Tensor, Nested[Tensor]]:
        """Computes decoder loss.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim] of encoder outputs.
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
                target_labels: An int Tensor of shape [batch_size, num_labels].
                target/input_ids: Optionally an int Tensor of shape [batch_size, num_labels].
                For both target_labels and target/input_ids, values should be in the range
                [0, vocab_size). Out-of-range values are excluded from the loss calculation
                (e.g., paddings and EOS can be represented this way).

        Returns:
            A tuple (loss, aux_outputs):
                loss: A scalar loss value.
                aux_outputs: A dict containing:
                    per_example_loss: A float Tensor of shape [batch_size].
                    per_example_weight: A float Tensor of shape [batch_size].
        """
        raise NotImplementedError(type(self))


class CTCDecoderModel(ASRDecoderModelBase):
    """CTC decoder model.

    CTC maps continuous sequences (e.g. speech embeddings) to "labelings", sequences over a finite
    vocab (with size `vocab_size`). The vocab does not have to contain EOS.
    Output sequences should be no longer than input sequences, and may possibly be shorter (e.g.
    after removing repeated tokens and/or "blanks", represented by `blank_token_id`).

    Reference:
    https://dl.acm.org/doi/10.1145/1143844.1143891
    """

    @config_class
    class Config(ASRDecoderModelBase.Config):
        """Configures CTCDecoderModel."""

        # Layer to map hidden state to vocab logits.
        lm_head: BaseLayer.Config = Linear.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("lm_head", cfg.lm_head.set(input_dim=cfg.dim, output_dim=cfg.vocab_size))

    def predict(self, input_batch: Nested[Tensor]) -> Tensor:
        """Computes logits.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.

        Returns:
            Logits of shape [batch_size, num_frames, vocab_size]. Logits corresponding to padding
            frames will be 0's. Note that the returned logits are not proper log probabilities, i.e.
            we have not subtracted the log-partition function.
        """
        inputs = input_batch["inputs"]
        paddings = input_batch["paddings"]
        logits = self.lm_head(inputs)
        return logits * (1 - paddings[..., None])

    def forward(
        self,
        input_batch: Nested[Tensor],
    ) -> Tuple[Tensor, Nested[Tensor]]:
        """Computes CTC loss.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
                target_labels: An int Tensor of shape [batch_size, num_labels].
                    Values should be in the range [0, vocab_size). We assume there are no BOS
                    tokens, and that sequences are not truncated. Out-of-range values are excluded
                    from the loss calculation (e.g., paddings and EOS can be represented this way).

        Returns:
            A tuple (loss, aux_outputs):
                loss: A scalar loss value.
                aux_outputs: A dict containing:
                    per_example_loss: A float Tensor of shape [batch_size].
                    per_example_weight: A float Tensor of shape [batch_size].
        """
        cfg: CTCDecoderModel.Config = self.config
        paddings: Tensor = input_batch["paddings"]
        target_labels: Tensor = input_batch["target_labels"]

        # Infer target_paddings from out-of-range labels.
        target_paddings = jnp.logical_or(cfg.vocab_size <= target_labels, target_labels < 0)

        # Compute CTC loss.
        logits = self.predict(input_batch)
        per_example_loss = optax.ctc_loss(
            logits=logits,
            logit_paddings=paddings,
            labels=target_labels,
            label_paddings=target_paddings,
            blank_id=cfg.blank_token_id,
        )

        # Drop examples with targets longer than inputs.
        per_example_weight = _is_valid_ctc_seq(
            paddings=paddings, target_labels=target_labels, target_paddings=target_paddings
        )
        per_example_weight = per_example_weight.astype(per_example_loss.dtype)

        # Compute weighted loss.
        loss = jnp.sum(per_example_loss * per_example_weight) / jnp.maximum(
            per_example_weight.sum(), 1
        )
        aux_outputs = dict(per_example_weight=per_example_weight, per_example_loss=per_example_loss)
        return loss, aux_outputs

    def _tokens_to_scores(
        self,
        input_batch: Nested[Tensor],
        *,
        num_decodes: int,
        logits_modifier: Optional[ConfigOr[LogitsToLogitsFn]] = None,
    ) -> Callable[[Tensor, Nested[Tensor]], Tuple[Tensor, Nested[Tensor]]]:
        """Returns a function that maps current token IDs and model state to next logits and updated
        state, to be used with decoding (see e.g. `beam_search_decode` or `sample_decode`).
        """
        paddings = input_batch["paddings"]
        logits_modifier = maybe_instantiate(logits_modifier)

        # [batch_size, num_frames, vocab_size].
        logits = self.predict(input_batch)
        if logits.dtype in (jnp.bfloat16, jnp.float16):
            # Cast for log softmax.
            logits = logits.astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits)
        # Mask out log probs at padding frames.
        log_probs += paddings[..., None] * NEG_INF
        # Extend log_probs by 1 step, so we can always decode up to `num_frames` non-EOS tokens.
        # [batch_size, num_frames + 1, vocab_size].
        log_probs = jnp.pad(log_probs, ((0, 0), (0, 1), (0, 0)), constant_values=NEG_INF)
        # Add a dummy EOS token:
        # eos_log_probs[b, t, :] = 0 if paddings_extended[b, t] else NEG_INF.
        paddings_extended = jnp.pad(paddings, ((0, 0), (0, 1)), constant_values=1)
        eos_log_probs = (1 - paddings_extended[:, :, None]) * NEG_INF
        # [batch_size, num_frames + 1, vocab_size + 1].
        log_probs = jnp.concatenate([log_probs, eos_log_probs], axis=-1)
        # Apply logits modifier after (e.g. if applying top-k, don't factor in padding scores).
        if logits_modifier:
            log_probs = logits_modifier(log_probs)

        def tokens_to_scores(
            token_ids: Tensor, state: Nested[Tensor]
        ) -> Tuple[Tensor, Nested[Tensor]]:
            # CTC assumes conditional independence between frames.
            del token_ids
            time_step = state["time_step"]
            # [batch_size, vocab_size].
            log_probs_t = log_probs[:, time_step, :]
            # [batch_size * num_decodes, vocab_size].
            log_probs_t = flatten_decoding_dim(
                add_decoding_dim(log_probs_t, num_decodes=num_decodes),
            )
            state["time_step"] = time_step + 1
            return log_probs_t, state

        return tokens_to_scores

    def beam_search_decode(
        self,
        input_batch: Nested[Tensor],
        num_decodes: int = 1,
        prefix_merger: Optional[PrefixMerger] = None,
    ) -> DecodeOutputs:
        """CTC beam search decoding with optional prefix merging.

        The output hypotheses will have blanks and repeats removed (via `_map_label_sequences`).

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
            num_decodes: Beam size.
            prefix_merger: An optional PrefixMerger to apply during decoding.

        Returns:
            DecodeOutputs, containing:
                raw_sequences: An int Tensor of shape [batch_size, num_decodes, num_frames].
                sequences: An int Tensor of shape [batch_size, num_decodes, num_frames].
                paddings: A 0/1 Tensor of shape [batch_size, num_decodes, num_frames].
                scores: A Tensor of shape [batch_size, num_decodes].

        Raises:
            ValueError: If max_decode_len is not None.
        """
        cfg: CTCDecoderModel.Config = self.config
        paddings: Tensor = input_batch["paddings"]
        # Add 1 so we can drop EOS while ensuring decodes can be up to `num_frames`.
        max_decode_len = paddings.shape[-1] + 1
        beam_search_outputs = beam_search_decode(
            inputs=jnp.zeros_like(paddings),
            time_step=jnp.zeros(paddings.shape[0], dtype=paddings.dtype),
            cache={"time_step": jnp.array(0)},
            tokens_to_scores=self._tokens_to_scores(input_batch, num_decodes=num_decodes),
            num_decodes=num_decodes,
            eos_id=cfg.vocab_size,  # Dummy EOS token.
            max_decode_len=max_decode_len,
            prefix_merger=prefix_merger,
        )
        return self._postprocess_outputs(
            sequences=beam_search_outputs.sequences,
            paddings=paddings,
            scores=beam_search_outputs.scores,
        )

    def sample_decode(
        self,
        input_batch: Nested[Tensor],
        *,
        num_decodes: int = 1,
        logits_modifier: Optional[ConfigOr[LogitsToLogitsFn]] = None,
    ) -> DecodeOutputs:
        """CTC sample decoding.

        The output hypotheses will have blanks and repeats removed (via `_map_label_sequences`).
        To perform greedy decoding, provide `top_k_logits(1)` as the logits modifier.

        Args:
            input_batch: See `beam_search_decode`.
            num_decodes: See `beam_search_decode`.
            logits_modifier: An optional logits modifier to apply prior to softmax.
                If None, do not modify the logits.

        Returns:
            See `beam_search_decode`.
        """
        cfg: CTCDecoderModel.Config = self.config
        paddings: Tensor = input_batch["paddings"]
        # Add 1 so we can drop EOS while ensuring decodes can be up to `num_frames`.
        max_decode_len = paddings.shape[-1] + 1
        sample_decode_outputs = sample_decode(
            inputs=jnp.zeros_like(paddings),
            time_step=jnp.zeros(paddings.shape[0], dtype=paddings.dtype),
            cache={"time_step": jnp.array(0)},
            tokens_to_scores=self._tokens_to_scores(
                input_batch, num_decodes=num_decodes, logits_modifier=logits_modifier
            ),
            num_decodes=num_decodes,
            prng_key=self.prng_key,
            max_decode_len=max_decode_len,
            stop_decoding_condition=StopOnSubsequence([[cfg.vocab_size]]),  # Dummy EOS token.
        )
        return self._postprocess_outputs(
            sequences=sample_decode_outputs.sequences,
            paddings=paddings,
            scores=sample_decode_outputs.token_scores,
        )

    def greedy_decode(self, input_batch: Nested[Tensor]) -> DecodeOutputs:
        """CTC greedy decoding.

        The output hypotheses will have blanks and repeats removed (via `_map_label_sequences`).

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.

        Returns:
            DecodeOutputs, containing:
                raw_sequences: An int Tensor of shape [batch_size, 1, num_frames].
                sequences: An int Tensor of shape [batch_size, 1, num_frames].
                paddings: A 0/1 Tensor of shape [batch_size, 1, num_frames].
                scores: A Tensor of shape [batch_size, 1].
        """
        cfg: CTCDecoderModel.Config = self.config
        paddings: Tensor = input_batch["paddings"]
        # [batch_size, num_frames, vocab_size].
        logits = self.predict(input_batch)
        # [batch, 1, num_frames].
        sequences = jnp.argmax(logits, axis=-1)[:, None, :]
        # Remove repeats and blanks.
        # We make the assumption that the trailing padding positions have 0 as the argmax index.
        outputs = _map_label_sequences(inputs=sequences, blank_id=cfg.blank_token_id, pad_id=0)

        # [batch_size, num_frames, vocab_size].
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_probs += paddings[..., None] * NEG_INF
        # [batch, num_frames, 1].
        scores = jnp.take_along_axis(log_probs, sequences[:, 0, :, None], axis=-1)
        # [batch, 1].
        scores = jnp.sum(jnp.squeeze(scores, axis=-1) * (1 - paddings), axis=1, keepdims=True)

        return DecodeOutputs(
            raw_sequences=sequences,
            sequences=outputs["sequences"],
            paddings=outputs["paddings"],
            scores=scores,
        )

    def _postprocess_outputs(self, *, sequences: Tensor, paddings: Tensor, scores: Tensor):
        cfg: CTCDecoderModel.Config = self.config
        live_mask = 1 - paddings[:, None, :]
        # Drop dummy decode position and mask outputs corresponding to padding frames.
        sequences = sequences[..., :-1] * live_mask
        # If given per-token scores, sum non-padding scores along sequence dim.
        if scores.ndim == 3:
            scores = (scores[..., :-1] * live_mask).sum(axis=-1)
        # Remove repeats and blanks.
        outputs = _map_label_sequences(inputs=sequences, blank_id=cfg.blank_token_id, pad_id=0)
        return DecodeOutputs(
            raw_sequences=sequences,
            sequences=outputs["sequences"],
            paddings=outputs["paddings"],
            scores=scores,
        )


def _map_label_sequences(inputs: Tensor, *, blank_id: int = 0, pad_id: int = 0) -> Nested[Tensor]:
    """Removes blanks, paddings, and repeats from the input sequences, as seen in CTC.

    Args:
        inputs: An int Tensor of shape [..., max_decode_len] containing decoded sequences.
        blank_id: Token ID corresponding to blanks.
        pad_id: Token ID corresponding to paddings.

    Returns:
        A dict containing:
            sequences: A Tensor of shape [..., max_decode_len] containing label sequences.
            paddings: A 0/1 Tensor of shape [..., max_decode_len]. 1's represent paddings.
            lengths: A Tensor of shape [..., 1] containing the length of each sequence.
    """
    max_decode_len = inputs.shape[-1]

    # Identify points at which curr != prev, excluding blanks and paddings.
    # `indicators` will have shape [batch_size, num_decodes, max_decode_len], and have a value
    # of 1 in positions corresponding to inputs we intend to keep (i.e., the token is not blank
    # or padding, and is different from the previous token).
    y = jnp.concatenate([jnp.full(inputs.shape[:-1] + (1,), pad_id), inputs], axis=-1)
    indicators = (y[..., 1:] != y[..., :-1]) & (inputs != blank_id) & (inputs != pad_id)

    # Compute lengths of final sequences. [..., 1].
    lens = jnp.sum(indicators, axis=-1, keepdims=True, dtype=inputs.dtype)

    # Compute sequences by left-justifying the tokens-to-keep. Under jit, we use a dispatch matrix
    # of shape [batch_size, num_decodes, max_decode_len, max_decode_len]. dispatch[..., i, j] == 1
    # means token i goes to position j. dispatch[..., i, :] == 0 means we drop token i.
    # [batch_size, num_decodes, max_decode_len, max_decode_len].
    dispatch = jax.nn.one_hot(
        jnp.cumsum(indicators, axis=-1) * indicators - 1, max_decode_len, dtype=inputs.dtype
    )
    sequences = jnp.einsum("...nm,...n->...m", dispatch, inputs)
    paddings = (jnp.arange(max_decode_len) >= lens).astype(inputs.dtype)
    if pad_id != 0:
        sequences = jnp.where(paddings, pad_id, sequences)
    return dict(sequences=sequences, paddings=paddings, lengths=lens)


def _remove_blank_tokens(inputs: Tensor, *, paddings: Tensor, blank_id: int = 0):
    """Removes blank tokens from the input sequences, as seen in RNN-T.

    Args:
        inputs: An int Tensor of shape [batch_size, num_decodes, max_decode_len] of sequences
            that contain blank tokens.
        paddings: A 0/1 Tensor of shape [batch_size, num_decodes, max_decode_len].
        blank_id: Token ID corresponding to blanks.

    Returns:
        A dict containing:
            sequences: A Tensor of shape [batch_size, num_decodes, max_decode_len] containing
                label sequences.
            paddings: A 0/1 Tensor of shape [batch_size, num_decodes, max_decode_len].
                1's represent paddings.
    """
    max_decode_len = inputs.shape[-1]
    # [batch, beam, seq].
    is_non_blank = (inputs != blank_id).astype(jnp.int32)
    # cum_non_blanks[i, k, t] = #(non-blanks in inputs[i, k, :t-1]),
    # if is_non_blank and (1-paddings) else -1.
    cum_non_blanks = jnp.cumsum(is_non_blank, axis=-1) * is_non_blank * (1 - paddings) - 1
    # [batch, beam, seq, seq].
    # dispatch[:, :, from, to] = 1 if inputs[:, :, from] is put at sequences[:, :, to].
    dispatch = jax.nn.one_hot(cum_non_blanks, max_decode_len).astype(jnp.int32)
    sequences = jnp.einsum("bkf,bkft->bkt", inputs, dispatch)
    # Compute lengths of final sequences. [..., 1].
    lens = jnp.max(cum_non_blanks, axis=-1, keepdims=True)
    paddings = (jnp.arange(max_decode_len)[None, None, :] > lens).astype(inputs.dtype)
    return dict(sequences=sequences, paddings=paddings)


class RNNPredictionNetwork(BaseLayer):
    """Rnn prediction network internal language model."""

    @config_class
    class Config(BaseLayer.Config):
        """Configs RNNPredictionNetwork."""

        # Vocab size.
        vocab_size: Required[int] = REQUIRED
        # The embedding dim.
        emb_dim: Required[int] = REQUIRED
        # The output dim.
        output_dim: Required[int] = REQUIRED

        # Embedding lookup layer.
        embedding: Embedding.Config = Embedding.default_config()
        # Rnn cell of the internal LM model. Defaults to a 1 layer LSTM.
        rnn_cell: BaseRNNCell.Config = LSTMCell.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "embedding", cfg.embedding.set(num_embeddings=cfg.vocab_size, dim=cfg.emb_dim)
        )
        rnn_cfg = cfg.rnn_cell.set(input_dim=cfg.emb_dim, output_dim=cfg.output_dim)
        self._add_child("rnn", rnn_cfg)

    def forward(self, inputs: Tensor) -> Tensor:
        time_major_outputs = self.rnn(
            time_major_inputs=jnp.transpose(self.embedding(x=inputs), [1, 0, 2])
        )
        return jnp.transpose(time_major_outputs, [1, 0, 2])

    def init_step_states(self, *, batch_size: int) -> NestedTensor:
        return self.rnn.init_step_states(batch_size=batch_size)

    def extend_step(
        self,
        *,
        inputs: NestedTensor,
        step_states: NestedTensor,
    ) -> Tuple[NestedTensor, NestedTensor]:
        return self.rnn.extend_step(inputs=self.embedding(x=inputs), step_states=step_states)


class TransducerDecoderModel(ASRDecoderModelBase):
    """Transducer decoder.

    It is often referred as rnn-transducer or rnnt in the literature.
    """

    @config_class
    class Config(ASRDecoderModelBase.Config):
        """Configures TransducerDecoderModel."""

        # The lm dim.
        lm_dim: Required[int] = REQUIRED
        # The joint network dim.
        joint_dim: Required[int] = REQUIRED

        bos_id: int = 1
        eos_id: int = 2

        # Prediction network internal language model.
        prediction_network: RNNPredictionNetwork.Config = RNNPredictionNetwork.default_config()
        # Joint network that combines acoustic model and language model features.
        # AM projection.
        am_proj: Linear.Config = Linear.default_config()
        # LM projection.
        lm_proj: Linear.Config = Linear.default_config()
        # Transducer that maps the hidden state to vocab logits.
        transducer: InstantiableConfig = Transducer.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.eos_id == 0:
            raise ValueError("Use a non-zero eos_id for the transducer model.")
        self.vlog(
            3,
            (
                f"am_dim={cfg.dim}, lm_dim={cfg.lm_dim}, joint_dim={cfg.joint_dim}，"
                f"vocab_size={cfg.vocab_size}."
            ),
        )
        # In most common cases, am_data and lm_data are summed together after the projection, thus
        # we only keep one bias in the two projections.
        self._add_child(
            "am_proj", cfg.am_proj.set(input_dim=cfg.dim, output_dim=cfg.joint_dim, bias=True)
        )
        self._add_child(
            "lm_proj", cfg.lm_proj.set(input_dim=cfg.lm_dim, output_dim=cfg.joint_dim, bias=False)
        )
        self._add_child(
            "prediction_network",
            cfg.prediction_network.set(
                vocab_size=cfg.vocab_size,
                output_dim=cfg.lm_dim,
            ),
        )
        self._add_child(
            "transducer",
            cfg.transducer.set(input_dim=cfg.joint_dim, vocab_size=cfg.vocab_size),
        )

    def forward(self, input_batch: NestedTensor) -> Tuple[Tensor, NestedTensor]:
        """Computes the transducer loss.

        Args:
            input_batch: A dict containing:
                - inputs: A Tensor of shape [batch_size, num_frames, dim].
                - paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
                - target_labels: an int Tensor of shape [batch_size, num_labels]. Prediction target
                    of the transducer decoder.
                - target/input_ids: an int Tensor of shape [batch_size, num_labels]. Prediction
                    inputs to the transducer decoder. It starts with

            For both target_labels and target/input_ids, values should be in the range
            [0, vocab_size). target_labels does not contain BOS and valid label tokens are
            followed by a EOS token. input_ids starts with a BOS token. Sequences are not
            truncated. Out-of-range values are excluded from the loss calculation.

        Returns:
            (loss, per_example), where `loss` is a scalar representing the transducer loss
            and `per_example` is a dict containing decoder output. It has the following keys:
            - "weight": [batch_size], the aggregation weight of the per-example loss.
            - "loss": [batch_size], per-example loss. Invalid example is masked out.
            aggregated_loss = sum(per_example["loss"] * per_example["weight"]) /
                sum(per_example["weight"]).
        """
        cfg = self.config
        # [batch, src_len, joint_dim].
        am_data = self.am_proj(input_batch["inputs"])
        am_paddings: Tensor = input_batch["paddings"]

        target_labels: Tensor = input_batch["target_labels"]
        # Infer target_paddings from out-of-range labels.
        target_paddings = jnp.logical_or(cfg.vocab_size <= target_labels, target_labels < 0)

        # [batch, tgt_len, joint_dim].
        lm_data = self.lm_proj(self.prediction_network(inputs=input_batch["target"]["input_ids"]))

        _, per_example = self.transducer(
            am_data=am_data,
            am_paddings=am_paddings,
            lm_data=lm_data,
            lm_paddings=target_paddings,
            target_labels=target_labels,
        )
        per_example_loss, per_example_weight = (
            per_example["loss"],
            per_example["is_valid_example"],
        )
        per_example_weight = per_example_weight.astype(per_example_loss.dtype)

        # Compute weighted loss.
        loss = jnp.sum(per_example_loss * per_example_weight) / jnp.maximum(
            per_example_weight.sum(), 1
        )
        aux_outputs = dict(per_example_weight=per_example_weight, per_example_loss=per_example_loss)
        return loss, aux_outputs

    def _tokens_to_scores(
        self,
        input_batch: Nested[Tensor],
        *,
        num_decodes: int,
        max_decode_len: int,
    ) -> Callable[[Tensor, Nested[Tensor]], Tuple[Tensor, Nested[Tensor]]]:
        """Returns a function that maps current token IDs and model state to next logits and updated
            state, to be used with decoding, see `beam_search_decode`.

        The signature is [batch*beam, vocab], {} = tokens_to_scores([batch*beam, 1], {}).
            state_cache contains keys:
            - am_step: the am frame index.
            - lm_states: the prediction network rnn states.
            - lm_data: the projected rnn prediction network outputs.
            - decode_step: number of decode steps.
        """
        cfg = self.config
        vocab_size = cfg.vocab_size
        blank_id, eos_id = cfg.blank_id, cfg.eos_id
        # [batch].
        src_len = jnp.sum(1 - input_batch["paddings"], axis=-1)
        # [batch, src_max_len, joint_dim].
        am_data = self.am_proj(input_batch["inputs"])
        batch_size, src_max_len = input_batch["paddings"].shape

        def tokens_to_scores(
            token_ids: Tensor, state_cache: NestedTensor
        ) -> Tuple[Tensor, NestedTensor]:
            # [batch*beam, 1].
            is_blank = token_ids == blank_id

            # 1. Computes am_data at current step.
            # [batch*beam].
            am_step_at_t_flatten = state_cache["am_step"] + jnp.squeeze(is_blank, axis=1)
            # [batch, beam].
            am_step_at_t = unflatten_decoding_dim(
                am_step_at_t_flatten, batch_size=batch_size, num_decodes=num_decodes
            )
            # [batch, beam, src_len].
            am_indices_at_t = jax.nn.one_hot(am_step_at_t, src_max_len, dtype=am_data.dtype)

            # Slice am_t. am_data_at_t[b, k, :] = am_data[b, am_step_at_t[b, k], :].
            # [batch, beam, joint_dim].
            am_data_at_t = jnp.einsum("bso,bks->bko", am_data, am_indices_at_t)
            # [batch*beam, 1, joint_dim]. Flatten and add back the sequence dimension.
            am_data_at_t = flatten_decoding_dim(am_data_at_t)[:, None, :]
            # 2. Computes lm_data at current step.
            with child_context("prediction_network_decode", module=self.prediction_network):
                # [batch*beam, ...], [batch*beam, joint_dim].
                new_lm_states, new_preproj_lm_data = self.prediction_network.extend_step(
                    inputs=jnp.squeeze(token_ids, axis=-1),
                    step_states=state_cache["lm_states"],
                )
            new_lm_data = self.lm_proj(new_preproj_lm_data)
            # lm_data = state_cache["lm_data"] if is_blank else new_lm_data.
            # [batch*beam, 1, joint_dim].
            lm_data_at_t = (
                state_cache["lm_data"] * is_blank[:, :, None]
                + (new_lm_data * (1 - is_blank))[:, None, :]
            )

            # updated_lm_states = state_cache["lm_states"] if is_blank else new_lm_states.
            # [batch*beam, ...].
            lm_states_at_t = vectorized_tree_map(
                lambda x1, x2: x1 * is_blank + x2 * (1 - is_blank),
                state_cache["lm_states"],
                new_lm_states,
            )
            pred = self.transducer.predict(am_data=am_data_at_t, lm_data=lm_data_at_t)

            # [batch*beam, 1, 1, vocab].
            log_probs = log_probs_from_blank_and_tokens(
                log_prob_blank=pred["log_prob_blank"],  # [batch*beam, 1, 1].
                log_prob_tokens=pred["log_prob_tokens"],  # [batch*beam, 1, 1, vocab].
                blank_id=blank_id,
            )
            # [batch*beam, vocab].
            log_probs = jnp.squeeze(log_probs, axis=(1, 2))

            # Force eos when all speech frames are consumed or at the last step.
            # [batch*beam, 1].
            force_eos = jnp.logical_or(
                # all frames are consumed.
                flatten_decoding_dim(am_step_at_t >= src_len[:, None]),
                # reaches last step
                state_cache["decode_step"] == max_decode_len - 1,
            )[:, None]

            # [1, vocab].
            eos_id_onehot = jax.nn.one_hot(eos_id, vocab_size, dtype=jnp.int32)[None, :]
            # log_probs[b, eos] = 0 if force_eos[b] else NEG_INF.
            # log_probs is of shape [batch*beam, vocab].
            log_probs *= 1 - eos_id_onehot
            log_probs += (1 - force_eos) * eos_id_onehot * NEG_INF
            # log_probs[b, non_eos] = NEG_INF if force_eos[b].
            log_probs += force_eos * (1 - eos_id_onehot) * NEG_INF

            new_cache = dict(
                am_step=am_step_at_t_flatten,
                lm_data=lm_data_at_t,
                lm_states=lm_states_at_t,
                decode_step=state_cache["decode_step"] + 1,
            )
            return log_probs, new_cache

        return tokens_to_scores

    def beam_search_decode(
        self,
        input_batch: Nested[Tensor],
        num_decodes: int,
        max_decode_len: int,
    ) -> DecodeOutputs:
        """Transducer label-synchronous search.

        Each hypothesis in the beam has the same length of tokens, including
            both blank and label tokens.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim] from encoder outputs.
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
            num_decodes: Beam size.
            max_decode_len: maximum number of decode steps to run beam search.
                Decoding terminates if an eos token is not emitted after max_decode_steps
                steps. This value can depend on the tokenization.

        Returns:
            DecodeOutputs, containing
                raw_sequences: An int Tensor of shape [batch_size, num_decodes, num_frames].
                sequences: An int Tensor of shape [batch_size, num_decodes, num_frames].
                paddings: A 0/1 Tensor of shape [batch_size, num_decodes, num_frames].
                scores: A Tensor of shape [batch_size, num_decodes].

        Raises:
            ValueError: If max_decode_len <= src_max_len.
        """
        batch_size, src_max_len = input_batch["paddings"].shape
        if max_decode_len <= src_max_len:
            raise ValueError(
                f"max_decode_len = {max_decode_len} is smaller than src_max_len={src_max_len}."
            )

        cfg = self.config
        blank_id, eos_id, bos_id = cfg.blank_id, cfg.eos_id, cfg.bos_id

        # Starts decoding with [BOS] token.
        inputs = jnp.zeros((batch_size, max_decode_len))
        inputs = inputs.at[:, 0].set(bos_id)

        init_step_states = {
            "am_step": jnp.zeros(batch_size),
            "lm_states": self.prediction_network.init_step_states(batch_size=batch_size),
            "lm_data": jnp.zeros((batch_size, 1, self.config.joint_dim)),
            "decode_step": jnp.array(0),
        }

        beam_search_outputs = beam_search_decode(
            inputs=inputs,
            time_step=infer_initial_time_step(inputs, pad_id=0),
            cache=init_step_states,
            tokens_to_scores=self._tokens_to_scores(
                input_batch, num_decodes=num_decodes, max_decode_len=max_decode_len
            ),
            eos_id=eos_id,
            num_decodes=num_decodes,
            max_decode_len=max_decode_len,
        )

        decode_paddings = jnp.logical_or(
            jnp.cumsum(beam_search_outputs.sequences == eos_id, axis=-1),
            # Return all paddings for invalid sequences.
            (beam_search_outputs.scores == NEG_INF)[..., None],
        )
        # Remove blanks.
        outputs = _remove_blank_tokens(
            inputs=beam_search_outputs.sequences,
            paddings=decode_paddings,
            blank_id=blank_id,
        )
        return DecodeOutputs(
            raw_sequences=beam_search_outputs.sequences,
            sequences=outputs["sequences"],
            paddings=outputs["paddings"],
            scores=beam_search_outputs.scores,
        )
