# Copyright 2021 Google Research. All Rights Reserved.
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
# ==============================================================================
"""Hparams for model architecture and trainer."""
import ast
import collections
import copy
from typing import Any, Dict, Text
import tensorflow as tf
import yaml


def eval_str_fn(val):
  if '|' in val:
    return [eval_str_fn(v) for v in val.split('|')]
  if val in {'true', 'false'}:
    return val == 'true'
  try:
    return ast.literal_eval(val)
  except (ValueError, SyntaxError):
    return val


# pylint: disable=protected-access
class Config(dict):
  """A config utility class."""

  def __init__(self, *args, **kwargs):
    super().__init__()
    input_config_dict = dict(*args, **kwargs)
    self.update(input_config_dict)

  def __len__(self):
    return len(self.__dict__)

  def __setattr__(self, k, v):
    if isinstance(v, dict) and not isinstance(v, Config):
      self.__dict__[k] = Config(v)
    else:
      self.__dict__[k] = copy.deepcopy(v)

  def __getattr__(self, k):
    return self.__dict__[k]

  def __setitem__(self, k, v):
    self.__setattr__(k, v)

  def __getitem__(self, k):
    return self.__dict__[k]

  def __iter__(self):
    for key in self.__dict__:
      yield key

  def items(self):
    for key, value in self.__dict__.items():
      yield key, value

  def __repr__(self):
    return repr(self.as_dict())

  def __getstate__(self):
    return self.__dict__

  def __copy__(self):
    cls = self.__class__
    result = cls.__new__(cls)
    result.__dict__.update(self.__dict__)
    return result

  def __deepcopy__(self, memo):
    cls = self.__class__
    result = cls.__new__(cls)
    for k, v in self.__dict__.items():
      result[k] = v
    return result

  def __str__(self):
    try:
      return yaml.dump(self.as_dict(), indent=4)
    except TypeError:
      return str(self.as_dict())

  def _update(self, config_dict, allow_new_keys=True):
    """Recursively update internal members."""
    if not config_dict:
      return

    for k, v in config_dict.items():
      if k not in self.__dict__:
        if allow_new_keys:
          self.__setattr__(k, v)
        else:
          raise KeyError('Key `{}` does not exist for overriding. '.format(k))
      else:
        if isinstance(self.__dict__[k], Config) and isinstance(v, dict):
          self.__dict__[k]._update(v, allow_new_keys)
        elif isinstance(self.__dict__[k], Config) and isinstance(v, Config):
          self.__dict__[k]._update(v.as_dict(), allow_new_keys)
        else:
          self.__setattr__(k, v)

  def get(self, k, default_value=None):
    return self.__dict__.get(k, default_value)

  def update(self, config_dict):
    """Update members while allowing new keys."""
    self._update(config_dict, allow_new_keys=True)

  def keys(self):
    return self.__dict__.keys()

  def override(self, config_dict_or_str, allow_new_keys=False):
    """Update members while disallowing new keys."""
    if not config_dict_or_str:
      return
    if isinstance(config_dict_or_str, str):
      if '=' in config_dict_or_str:
        config_dict = self.parse_from_str(config_dict_or_str)
      elif config_dict_or_str.endswith('.yaml'):
        config_dict = self.parse_from_yaml(config_dict_or_str)
      else:
        raise ValueError(
            'Invalid string {}, must end with .yaml or contains "=".'.format(
                config_dict_or_str))
    elif isinstance(config_dict_or_str, dict):
      config_dict = config_dict_or_str
    else:
      raise ValueError('Unknown value type: {}'.format(config_dict_or_str))

    self._update(config_dict, allow_new_keys)

  def parse_from_yaml(self, yaml_file_path: Text) -> Dict[Any, Any]:
    """Parses a yaml file and returns a dictionary."""
    with tf.io.gfile.GFile(yaml_file_path, 'r') as f:
      config_dict = yaml.load(f, Loader=yaml.FullLoader)
      return config_dict

  def save_to_yaml(self, yaml_file_path):
    """Write a dictionary into a yaml file."""
    with tf.io.gfile.GFile(yaml_file_path, 'w') as f:
      yaml.dump(self.as_dict(), f, default_flow_style=False)

  def parse_from_str(self, config_str: Text) -> Dict[Any, Any]:
    """Parse a string like 'x.y=1,x.z=2' to nested dict {x: {y: 1, z: 2}}."""
    if not config_str:
      return {}
    config_dict = {}
    try:
      for kv_pair in config_str.split(','):
        if not kv_pair:  # skip empty string
          continue
        key_str, value_str = kv_pair.split('=')
        key_str = key_str.strip()

        def add_kv_recursive(k, v):
          """Recursively parse x.y.z=tt to {x: {y: {z: tt}}}."""
          if '.' not in k:
            return {k: eval_str_fn(v)}
          pos = k.index('.')
          return {k[:pos]: add_kv_recursive(k[pos + 1:], v)}

        def merge_dict_recursive(target, src):
          """Recursively merge two nested dictionary."""
          for k in src.keys():
            if ((k in target and isinstance(target[k], dict) and
                 isinstance(src[k], collections.abc.Mapping))):
              merge_dict_recursive(target[k], src[k])
            else:
              target[k] = src[k]

        merge_dict_recursive(config_dict, add_kv_recursive(key_str, value_str))
      return config_dict
    except ValueError:
      raise ValueError('Invalid config_str: {}'.format(config_str))

  def as_dict(self):
    """Returns a dict representation."""
    config_dict = {}
    for k, v in self.__dict__.items():
      if isinstance(v, Config):
        config_dict[k] = v.as_dict()
      elif isinstance(v, (list, tuple)):
        config_dict[k] = [
            i.as_dict() if isinstance(i, Config) else copy.deepcopy(i)
            for i in v
        ]
      else:
        config_dict[k] = copy.deepcopy(v)
    return config_dict
    # pylint: enable=protected-access


registry_map = {}


def register(cls, prefix='effnet:'):
  """Register a function, mainly for config here."""
  registry_map[prefix + cls.__name__.lower()] = cls
  return cls


def lookup(name, prefix='effnet:') -> Any:
  name = prefix + name.lower()
  if name not in registry_map:
    raise ValueError(f'{name} not registered: {registry_map.keys()}')
  return registry_map[name]


# needed?
# --params_override 
# --arch or model_name


base_config = Config(
    # model related params.
    model=Config(), # must be provided in full via model cfg files
    
    # train related params.
    train=Config(
        
        img_size=224,
        max_epochs=300, 
        steps_per_epoch=None, 
        batch_size=32, # renamed from train_batch_size
        use_dali=0, 
        
        # optimizer
        optimizer='rmsprop', 
        momentum=0.9, # rmsprop, momentum opt
        beta_1=0.0, # for adam.adamw
        beta_2=0.0, # for adam,adamw
        nesterov=0, # for sgd, momentum opt
        epsilon=.001, # for adamw, adam, rmsprop
        decay=0.9, #  for rmsprop
        # While the original implementation used a weight decay of 1e-5,
        # tf.nn.l2_loss divides it by 2, so we halve this to compensate in Keras
        weight_decay=5e-6, # for adamw or can be used in learnable layers as L2 reg.
        label_smoothing=0.1, 
        # The optimizer iteratively updates two sets of weights: the search directions for weights
        # are chosen by the inner optimizer, while the "slow weights" are updated each k steps 
        # based on the directions of the "fast weights" and the two sets of weights are 
        # synchronized. This method improves the learning stability and lowers the variance of
        # its inner optimizer.
        lookahead=0, # binary
        # Empirically it has been found that using the moving average of the trained parameters
        # of a deep network is better than using its trained parameters directly. This optimizer
        # allows you to compute this moving average and swap the variables at save time so that
        # any code outside of the training loop will use by default the average values instead
        # of the original ones.
        moving_average_decay=0.0,
        # model evaluation during training can be done using the original weights
        # or using EMA weights. The latter takes place if moving_average_decay > 0 and intratrain_eval_using_ema is True)
        intratrain_eval_using_ema=True,
        # to simulate a large batch size
        grad_accum_steps=1,
        # grad clipping is used in the custom train_step, which is called when grad_accum_steps > 1
        grad_clip_norm=0,
        # to optimize grad reducing across all workers
        hvd_fp16_compression = True,
        create_SavedModel=False,

        #lr schedule
        lr_decay='exponential',
        lr_init=0.008, 
        lr_decay_epochs=2.4, 
        lr_decay_rate=0.97, 
        lr_warmup_epochs=5, 
        
        # metrics
        metrics =  ['accuracy', 'top_5'], # used in tr and eval
        
        # load and save ckpt
        resume_checkpoint=1, # binary
        save_checkpoint_freq=5, 
        
        # progressive training (active when n_stages>1)
        n_stages=1, # progressive tr
        base_img_size=128,  
        base_mixup=0,
        base_cutmix=0,
        base_randaug_mag=5,
        
        #callbacks
        enable_checkpoint_and_export=1, # binary
        enable_tensorboard=0, # binary
        tb_write_model_weights=0, # tb: tensorboard, binary
    ),
    eval=Config(
        skip_eval=0, # binary
        num_epochs_between_eval=1, 
        use_dali=0, # binary, renamed from use_dali_eval
        batch_size=100, # for accurate eval, it should divide the number of validation samples 
        img_size=224,
        export=0
    ),
    predict=Config(
        ckpt=None, # renamed from inference_checkpoint
        img_dir='/infer_data/', # renamed from to_predict
        batch_size=32, # renamed from predict_batch_size
        img_size=224,
        benchmark=0, 
    ),
    # data related params.
    data=Config(
        dataset='ImageNet', 
        augmenter_name='autoaugment',  
        
        #Rand-augment params
        raug_num_layers=None, 
        raug_magnitude=None, 
        cutout_const=None, 
        mixup_alpha=0., 
        cutmix_alpha=0., 
        defer_img_mixing=True,
        translate_const=None, 
        
        #Auto-augment params
        autoaugmentation_name=None, 
        
        # used in dali
        index_file='', 
        
        #dataset and split
        data_dir='/data/', 
        num_classes=1000, # must match the one in model config
        train_num_examples=1281167, 
        eval_num_examples=50000, 
        
        # image normalization
        mean_subtract_in_dpipe=False, 
        standardize_in_dpipe=False,
        
        # Set to False for 1-GPU training
        map_parallelization=True
    ),
    runtime=Config(

        use_amp=1, # binary
        log_steps=100, 
        mode='tran_and_eval', #OK
        time_history=1, # binary
        use_xla=1, # binary
        intraop_threads='', 
        interop_threads='', 
        model_dir='/results/', # ckpts
        log_filename='log.json',
        display_every=10, 
        seed=None,
        data_format='channels_first',
        run_eagerly=0, # binary
        memory_limit=None, ##set max memory that can be allocated by TF to avoid hanging
    ))