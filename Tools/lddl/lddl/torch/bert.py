import logging
import numpy as np
import os
import random
import torch
import transformers
from typing import List, Optional, Tuple, Union
from collections import deque

from lddl.utils import (get_all_parquets_under, get_all_bin_ids,
                        get_file_paths_for_bin_id, deserialize_np_array)
from .dataloader import Binned, DataLoader
from .datasets import ParquetDataset
from .log import DatasetLogger
from .utils import get_node_rank, get_nproc_per_node


def _decode_record_batch(b):
  b = b.to_pydict()
  if 'masked_lm_positions' in b:
    assert 'masked_lm_labels' in b
  columns = tuple((b[k] for k in (
      'A',
      'B',
      'is_random_next',
      'masked_lm_positions',
      'masked_lm_labels',
  ) if k in b))
  for sample in zip(*columns):
    yield sample


class BertPretrainDataset(ParquetDataset):

  def _decode_record_batch(self, b):
    return _decode_record_batch(b)


class BertPretrainBinned(Binned):

  def _get_batch_size(self, batch):
    return batch['input_ids'].size(0)


def _to_encoded_inputs(
    batch,
    tokenizer,
    sequence_length_alignment=8,
    ignore_index=-1,
):
  batch_size = len(batch)
  As, Bs, are_random_next = [], [], []
  static_masking = (len(batch[0]) > 3)
  if static_masking:
    assert len(batch[0]) == 5
    all_masked_lm_positions, all_masked_lm_labels = [], []
  # Unpack each field.
  for sample in batch:
    As.append(tuple(sample[0].split()))
    Bs.append(tuple(sample[1].split()))
    are_random_next.append(sample[2])
    if static_masking:
      all_masked_lm_positions.append(
          torch.from_numpy(deserialize_np_array(sample[3]).astype(int)))
      all_masked_lm_labels.append(sample[4].split())
  # Figure out the sequence length of this batch.
  batch_seq_len = max(
      (len(tokens_A) + len(tokens_B) + 3 for tokens_A, tokens_B in zip(As, Bs)))
  # Align the batch_seq_len to a multiple of sequence_length_alignment, because
  # TC doesn't like it otherwise.
  batch_seq_len = (((batch_seq_len - 1) // sequence_length_alignment + 1) *
                   sequence_length_alignment)
  # Allocate the input torch.Tensor's.
  input_ids = torch.zeros(batch_size, batch_seq_len, dtype=torch.long)
  token_type_ids = torch.zeros_like(input_ids)
  attention_mask = torch.zeros_like(input_ids)
  if static_masking:
    labels = torch.full_like(input_ids, ignore_index)
  else:
    special_tokens_mask = torch.zeros_like(input_ids)
  # Fill in the input torch.Tensor's.
  for sample_idx in range(batch_size):
    tokens_A, tokens_B = As[sample_idx], Bs[sample_idx]
    # Prepare the input token IDs.
    tokens = ('[CLS]',) + tokens_A + ('[SEP]',) + tokens_B + ('[SEP]',)
    input_ids[sample_idx, :len(tokens)] = torch.as_tensor(
        tokenizer.convert_tokens_to_ids(tokens),
        dtype=torch.long,
    )
    # Prepare the token type ids (segment ids).
    start_idx = len(tokens_A) + 2
    end_idx = len(tokens_A) + len(tokens_B) + 3
    token_type_ids[sample_idx, start_idx:end_idx] = 1
    # Prepare the attention mask (input mask).
    attention_mask[sample_idx, :end_idx] = 1
    if static_masking:
      # Prepare the MLM labels.
      labels[sample_idx, all_masked_lm_positions[sample_idx]] = torch.as_tensor(
          tokenizer.convert_tokens_to_ids(all_masked_lm_labels[sample_idx]),
          dtype=torch.long,
      )
    else:
      # Prepare special_tokens_mask (for DataCollatorForLanguageModeling)
      special_tokens_mask[sample_idx, 0] = 1
      special_tokens_mask[sample_idx, len(tokens_A) + 1] = 1
      special_tokens_mask[sample_idx, len(tokens_A) + len(tokens_B) + 2:] = 1
  # Compose output dict.
  encoded_inputs = {
      'input_ids':
          input_ids,
      'token_type_ids':
          token_type_ids,
      'attention_mask':
          attention_mask,
      'next_sentence_labels':
          torch.as_tensor(
              are_random_next,
              dtype=torch.long,
          ),
  }
  if static_masking:
    encoded_inputs['labels'] = labels
  else:
    encoded_inputs['special_tokens_mask'] = special_tokens_mask
  return encoded_inputs


def _mask_tokens(
    inputs,
    special_tokens_mask=None,
    tokenizer=None,
    mlm_probability=0.15,
    ignore_index=-1,
):
  """
  Prepare masked tokens inputs/labels for masked language modeling: 80% MASK,
  10% random, 10% original.
  """
  labels = inputs.clone()
  # We sample a few tokens in each sequence for MLM training (with probability
  # `mlm_probability`)
  probability_matrix = torch.full(labels.shape, mlm_probability)
  if special_tokens_mask is None:
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
  else:
    special_tokens_mask = special_tokens_mask.bool()

  probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
  masked_indices = torch.bernoulli(probability_matrix).bool()
  # We only compute loss on masked tokens
  labels[~masked_indices] = ignore_index

  # 80% of the time, we replace masked input tokens with tokenizer.mask_token
  # ([MASK])
  indices_replaced = (torch.bernoulli(torch.full(labels.shape, 0.8)).bool() &
                      masked_indices)
  inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
      tokenizer.mask_token)

  # 10% of the time, we replace masked input tokens with random word
  indices_random = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool() &
                    masked_indices & ~indices_replaced)
  random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
  inputs[indices_random] = random_words[indices_random]

  # The rest of the time (10% of the time) we keep the masked input tokens
  # unchanged
  return inputs, labels


def get_bert_pretrain_data_loader(
    path,
    local_rank=0,
    shuffle_buffer_size=16384,
    shuffle_buffer_warmup_factor=16,
    tokenizer_class=transformers.BertTokenizerFast,
    vocab_file=None,
    tokenizer_kwargs={},
    data_loader_class=DataLoader,
    data_loader_kwargs={},
    mlm_probability=0.15,
    base_seed=12345,
    log_dir=None,
    log_level=logging.INFO,
    return_raw_samples=False,
    start_epoch=0,
    sequence_length_alignment=8,
    ignore_index=-1,
):
  """Gets a PyTorch DataLoader for the BERT pretraining task.

  The LDDL DataLoader can be used in the same way as a normal PyTorch
  DataLoader. The 'persistent_workers' attribute will always be enabled.

  The LDDL DataLoader streams samples from disk into memory, and uses a shuffle
  buffer to perform shuffling: at each iteration, a random sample from the
  shuffle buffer is popped, and a new sample is pushed into the shuffle buffer
  at this vacant location.

  Args:
    path: A string of the path pointing to the directory that contains the
      pretraining dataset in the format of balanced parquet shards.
    local_rank: The local rank ID (on this node) of the current pretraining
      process.
    shuffle_buffer_size: The size of the shuffle buffer.
    shuffle_buffer_warmup_factor: At the beginning, the shuffle buffer is empty.
      Therefore, in order to fill the shuffle buffer, at each iteration, more
      samples need to be pushed into the shuffle buffer than being popped out
      of. This factor indicates how many samples is pushed into the shuffle
      buffer per 1 sample being popped out of the shuffle buffer, until the
      shuffle buffer is full.
    tokenizer_class: The HuggingFace tokenizer class for BERT pretraining.
    vocab_file: The path to a vocab file, or the name of a pretrained model
      registered on huggingface.co (e.g., 'bert-large-uncased') of which the
      vocab file is downloaded.
    tokenizer_kwargs: The arguments to the tokenizer class.
    data_loader_class: The class of the DataLoader.
    data_loader_kwargs: The arguments to the DataLoader class.
    mlm_probability: The probability for masking tokens in the masked language
      modeling task (in BERT pretraining).
    base_seed: A base seed value on which other seeds used in the DataLoader are
      based.
    log_dir: The path to a directory to store the logs from the LDDL DataLoader.
    log_level: The logging verbose level.
    return_raw_samples: If True, returns the raw string pairs instead of token
      indices.
    start_epoch: The epoch number to start from. An epoch is defined as going
      through every sample in a dataset once.
    sequence_length_alignment: To get the input tensors of token indices, each
      sequence in a batch will only be padded to the longest sequence in this
      batch. However, certain hardware features might prefer the shapes of the
      input tensors to meet certain conditions. For example, it's better for the
      Tensor Core on NVIDIA GPUs if the dimensions of the input tensors are
      divisible by 8. Therefore, this argument is an alignment factor such that
      the sequences in a batch will be padded to the first sequence length
      larger than the longest sequence in this batch and also divisible by this
      alignment factor.
    ignore_index: The label value for the unmasked tokens in the language
      modeling task (in BERT pretraining).

  Returns:
    A PyTorch DataLoader that, in each iteration, yield:
    - If return_raw_samples is False, a dict of 5 key-value pairs which are the
      necessary input for BERT pretraining:
      {
        'input_ids': a torch.Tensor of size [batch_size, sequence_length],
        'token_type_ids': a torch.Tensor of size [batch_size, sequence_length],
        'attention_mask': a torch.Tensor of size [batch_size, sequence_length],
        'labels': a torch.Tensor of size [batch_size, sequence_length],
        'next_sentence_labels': a torch.Tensor of size [batch_size],
      }
    - If return_raw_samples is True, a list of the following lists:
      [
        strings of the first sequences in the sequence pairs,
        strings of the second sequences in the sequence pairs,
        bools that indicate whether the second sequences are the next sequences
          for the first sequences,
        numpy.ndarrays of positions of the masked tokens for the masked language
          modeling task (only exists if static masking is enabled),
        strings of space-seperated labels of the masked tokens for the masked
          language modeling task (only exists if static masking is enabled),
      ]

  Examples:
    train_dataloader = lddl.torch.get_bert_pretrain_data_loader(
      input_dir,
      local_rank=local_rank,
      vocab_file=vocab_file,
      data_loader_kwargs={
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
      },
      log_level=logging.WARNING,
      start_epoch=start_epoch,
    )

    for epoch in range(start_epoch, start_epoch + epochs):
      for i, batch in enumerate(train_dataloader):
        prediction_scores, seq_relationship_score = model(
          input_ids=batch['input_ids'].to(device),
          token_type_ids=batch['token_type_ids'].to(device),
          attention_mask=batch['attention_mask'].to(device),
      )
      loss = criterion(
          prediction_scores,
          seq_relationship_score,
          batch['labels'].to(device),
          batch['next_sentence_labels'].to(device),
      )
      ...
  """
  assert isinstance(path, str)
  assert isinstance(local_rank, int) and local_rank >= 0
  assert isinstance(shuffle_buffer_size, int) and shuffle_buffer_size > 0
  assert (isinstance(shuffle_buffer_warmup_factor, int) and
          shuffle_buffer_warmup_factor > 0)
  assert tokenizer_class in {
      transformers.BertTokenizerFast, transformers.BertTokenizer
  }
  assert isinstance(vocab_file, str)
  assert isinstance(tokenizer_kwargs, dict)
  assert data_loader_class in {DataLoader}
  assert isinstance(data_loader_kwargs, dict)
  assert isinstance(mlm_probability, (int, float)) and 0 <= mlm_probability <= 1
  assert isinstance(base_seed, int)
  assert log_dir is None or isinstance(log_dir, str)
  assert log_level in {
      logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING,
      logging.ERROR, logging.CRITICAL
  }
  assert isinstance(return_raw_samples, bool)
  assert isinstance(start_epoch, int)

  if os.path.isfile(vocab_file):
    tokenizer = tokenizer_class(vocab_file, **tokenizer_kwargs)
  else:
    tokenizer = tokenizer_class.from_pretrained(vocab_file, **tokenizer_kwargs)

  def _batch_preprocess(batch):
    with torch.no_grad():
      encoded_inputs = _to_encoded_inputs(
          batch,
          tokenizer,
          sequence_length_alignment=sequence_length_alignment,
          ignore_index=ignore_index,
      )
      if 'special_tokens_mask' in encoded_inputs:  # Dynamic masking.
        special_tokens_mask = encoded_inputs.pop('special_tokens_mask', None)
        (encoded_inputs['input_ids'], encoded_inputs['labels']) = _mask_tokens(
            encoded_inputs['input_ids'],
            special_tokens_mask=special_tokens_mask,
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            ignore_index=ignore_index,
        )
    return encoded_inputs

  logger = DatasetLogger(
      log_dir=log_dir,
      node_rank=get_node_rank(nproc_per_node=get_nproc_per_node(local_rank)),
      local_rank=local_rank,
      log_level=log_level,
  )
  dataset_kwargs = {
      'local_rank': local_rank,
      'shuffle_buffer_size': shuffle_buffer_size,
      'shuffle_buffer_warmup_factor': shuffle_buffer_warmup_factor,
      'base_seed': base_seed,
      'logger': logger,
      'start_epoch': start_epoch,
  }

  extra_collate = data_loader_kwargs.get('collate_fn', lambda x: x)
  if not return_raw_samples:
    data_loader_kwargs['collate_fn'] = lambda batch: extra_collate(
        _batch_preprocess(batch))
  data_loader_kwargs['persistent_workers'] = True

  # Find all the parquet file paths and figure out whether it is binned or
  # un-binned.
  all_file_paths = get_all_parquets_under(path)
  bin_ids = get_all_bin_ids(all_file_paths)
  if len(bin_ids) > 0:
    data_loader = BertPretrainBinned(
        [
            data_loader_class(
                BertPretrainDataset(
                    get_file_paths_for_bin_id(all_file_paths, bin_id),
                    **dataset_kwargs,
                ),
                **data_loader_kwargs,
            ) for bin_id in bin_ids
        ],
        base_seed=base_seed,
        start_epoch=start_epoch,
        logger=logger,
    )
  else:  # un-binned
    data_loader = data_loader_class(
        BertPretrainDataset(all_file_paths, **dataset_kwargs),
        **data_loader_kwargs,
    )

  return data_loader
