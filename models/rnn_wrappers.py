import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from .modules import prenet


class DecoderPrenetWrapper(RNNCell):
  '''Runs RNN inputs through a prenet before sending them to the cell.'''
  def __init__(self, cell, is_training, layer_sizes, embed_to_concat):
    super(DecoderPrenetWrapper, self).__init__()
    self._cell = cell
    self._is_training = is_training
    self._layer_sizes = layer_sizes
    self._embed_to_concat = embed_to_concat

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def call(self, inputs, state):
    prenet_out = prenet(inputs, self._is_training, self._layer_sizes, scope='decoder_prenet')
    if self._embed_to_concat is not None:
        concat_out = tf.concat([prenet_out, self._embed_to_concat], axis=-1, name='speaker_concat')
        return self._cell(concat_out, state)

    return self._cell(prenet_out, state)

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)



class ConcatOutputAndAttentionWrapper(RNNCell):
  '''Concatenates RNN cell output with the attention context vector.

  This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
  attention_layer_size=None and output_attention=False. Such a cell's state will include an
  "attention" field that is the context vector.
  '''
  def __init__(self, cell, embed_to_concat):
    super(ConcatOutputAndAttentionWrapper, self).__init__()
    self._cell = cell
    self._embed_to_concat=embed_to_concat

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size + self._cell.state_size.attention

  def call(self, inputs, state):
    output, res_state = self._cell(inputs, state)
    tensors = [output, res_state.attention]
    if self._embed_to_concat is not None:
        tensors.append(self._embed_to_concat)
    return tf.concat(tensors, axis=-1), res_state

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)
