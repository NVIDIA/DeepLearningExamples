# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from omegaconf import OmegaConf

def default_detection_model_configs():
    """Returns a default detection configs."""
    h = OmegaConf.create()

    # model name.
    h.name = 'tf_efficientdet_d1'

    h.backbone_name = 'tf_efficientnet_b1'
    h.backbone_args = None  # FIXME sort out kwargs vs config for backbone creation

    # model specific, input preprocessing parameters
    h.image_size = 640

    # dataset specific head parameters
    h.num_classes = 90

    # feature + anchor config
    h.min_level = 3
    h.max_level = 7
    h.num_levels = h.max_level - h.min_level + 1
    h.num_scales = 3
    h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    h.anchor_scale = 4.0

    # FPN and head config
    h.pad_type = 'same'  # original TF models require an equivalent of Tensorflow 'SAME' padding
    h.act_type = 'swish'
    h.box_class_repeats = 3
    h.fpn_cell_repeats = 3
    h.fpn_channels = 88
    h.separable_conv = True
    h.apply_bn_for_resampling = True
    h.conv_after_downsample = False
    h.conv_bn_relu_pattern = False
    h.use_native_resize_op = False
    h.pooling_type = None
    h.redundant_bias = True  # original TF models have back to back bias + BN layers, not necessary!

    h.fpn_name = None
    h.fpn_config = None
    h.fpn_drop_path_rate = 0.  # No stochastic depth in default.

    # classification loss (used by train bench)
    h.alpha = 0.25
    h.gamma = 1.5

    # localization loss (used by train bench)
    h.delta = 0.1
    h.box_loss_weight = 50.0

    return h


backbone_config = {
    "efficientnet_b0": {
        "width_coeff": 1,
        "depth_coeff": 1,
        "resolution": 224,
        "dropout": 0.2,
        "checkpoint_path": "./jocbackbone_statedict.pth"
    },
    "efficientnet_b1": {
        "width_coeff": 1,
        "depth_coeff": 1.1,
        "resolution": 240,
        "dropout": 0.2,
        "checkpoint_path": ""
    },
    "efficientnet_b2": {
        "width_coeff": 1.1,
        "depth_coeff": 1.2,
        "resolution": 260,
        "dropout": 0.3,
        "checkpoint_path": ""
    },
    "efficientnet_b3": {
        "width_coeff": 1.2,
        "depth_coeff": 1.4,
        "resolution": 300,
        "dropout": 0.3,
        "checkpoint_path": ""
    },
    "efficientnet_b4": {
        "width_coeff": 1.4,
        "depth_coeff": 1.8,
        "resolution": 380,
        "dropout": 0.4,
        "checkpoint_path": "./jocbackbone_statedict_B4.pth"
    },
    "efficientnet_b5": {
        "width_coeff": 1.6,
        "depth_coeff": 2.2,
        "resolution": 456,
        "dropout": 0.4,
        "checkpoint_path": ""
    },
    "efficientnet_b6": {
        "width_coeff": 1.8,
        "depth_coeff": 2.6,
        "resolution": 528,
        "dropout": 0.5,
        "checkpoint_path": ""
    },
    "efficientnet_b7": {
        "width_coeff": 2.0,
        "depth_coeff": 3.1,
        "resolution": 600,
        "dropout": 0.5,
        "checkpoint_path": ""
    },
}


efficientdet_model_param_dict = dict(
    # Models with PyTorch friendly padding and my PyTorch pretrained backbones, training TBD
    efficientdet_d0=dict(
        name='efficientdet_d0',
        backbone_name='efficientnet_b0',
        image_size=512,
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.1),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d0-f3276ba8.pth',
    ),
    efficientdet_d1=dict(
        name='efficientdet_d1',
        backbone_name='efficientnet_b1',
        image_size=640,
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d1-bb7e98fe.pth',
    ),
    efficientdet_d2=dict(
        name='efficientdet_d2',
        backbone_name='efficientnet_b2',
        image_size=768,
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.2),
        url='',  # no pretrained weights yet
    ),
    efficientdet_d3=dict(
        name='efficientdet_d3',
        backbone_name='efficientnet_b3',
        image_size=896,
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.2),
        url='',  # no pretrained weights yet
    ),
    efficientdet_d4=dict(
        name='efficientdet_d4',
        backbone_name='efficientnet_b4',
        image_size=1024,
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url='',
    ),

    # My own experimental configs with alternate models, training TBD
    # Note: any 'timm' model in the EfficientDet family can be used as a backbone here.
    efficientdet_w0=dict(
        name='efficientdet_w0',  # 'wide'
        backbone_name='efficientnet_b0',
        image_size=512,
        fpn_channels=80,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(
            drop_path_rate=0.1,
            feature_location='depthwise'),  # features from after DW/SE in IR block
        url='',  # no pretrained weights yet
    ),
    mixdet_m=dict(
        name='mixdet_m',
        backbone_name='mixnet_m',
        image_size=512,
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.1),
        url='',  # no pretrained weights yet
    ),
    mixdet_l=dict(
        name='mixdet_l',
        backbone_name='mixnet_l',
        image_size=640,
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.2),
        url='',  # no pretrained weights yet
    ),
    mobiledetv2_110d=dict(
        name='mobiledetv2_110d',
        backbone_name='mobilenetv2_110d',
        image_size=384,
        fpn_channels=48,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        act_type='relu6',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.05),
        url='',  # no pretrained weights yet
    ),
    mobiledetv2_120d=dict(
        name='mobiledetv2_120d',
        backbone_name='mobilenetv2_120d',
        image_size=512,
        fpn_channels=56,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        act_type='relu6',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.1),
        url='',  # no pretrained weights yet
    ),
    mobiledetv3_large=dict(
        name='mobiledetv3_large',
        backbone_name='mobilenetv3_large_100',
        image_size=512,
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        act_type='hard_swish',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.1),
        url='',  # no pretrained weights yet
    ),

    # Models ported from Tensorflow with pretrained backbones ported from Tensorflow
    tf_efficientdet_d0=dict(
        name='tf_efficientdet_d0',
        backbone_name='tf_efficientnet_b0',
        image_size=512,
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0-d92fd44f.pth',
    ),
    tf_efficientdet_d1=dict(
        name='tf_efficientdet_d1',
        backbone_name='tf_efficientnet_b1',
        image_size=640,
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d1-4c7ebaf2.pth'
    ),
    tf_efficientdet_d2=dict(
        name='tf_efficientdet_d2',
        backbone_name='tf_efficientnet_b2',
        image_size=768,
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d2-cb4ce77d.pth',
    ),
    tf_efficientdet_d3=dict(
        name='tf_efficientdet_d3',
        backbone_name='tf_efficientnet_b3',
        image_size=896,
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3-b0ea2cbc.pth',
    ),
    tf_efficientdet_d4=dict(
        name='tf_efficientdet_d4',
        backbone_name='tf_efficientnet_b4',
        image_size=1024,
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d4-5b370b7a.pth',
    ),
    tf_efficientdet_d5=dict(
        name='tf_efficientdet_d5',
        backbone_name='tf_efficientnet_b5',
        image_size=1280,
        fpn_channels=288,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5-ef44aea8.pth',
    ),
    tf_efficientdet_d6=dict(
        name='tf_efficientdet_d6',
        backbone_name='tf_efficientnet_b6',
        image_size=1280,
        fpn_channels=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d6-51cb0132.pth'
    ),
    tf_efficientdet_d7=dict(
        name='tf_efficientdet_d7',
        backbone_name='tf_efficientnet_b6',
        image_size=1536,
        fpn_channels=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        anchor_scale=5.0,
        fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7_53-6d1d7a95.pth'
    ),

    # The lite configs are in TF automl repository but no weights yet and listed as 'not final'
    tf_efficientdet_lite0=dict(
        name='tf_efficientdet_lite0',
        backbone_name='tf_efficientnet_lite0',
        image_size=512,
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        act_type='relu',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.1),
        # unlike other tf_ models, this was not ported from tf automl impl, but trained from tf pretrained efficient lite
        # weights using this code, will likely replace if/when official det-lite weights are released
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_lite0-f5f303a9.pth',
    ),
    tf_efficientdet_lite1=dict(
        name='tf_efficientdet_lite1',
        backbone_name='tf_efficientnet_lite1',
        image_size=640,
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        act_type='relu',
        backbone_args=dict(drop_path_rate=0.2),
        url='',  # no pretrained weights yet
    ),
    tf_efficientdet_lite2=dict(
        name='tf_efficientdet_lite2',
        backbone_name='tf_efficientnet_lite2',
        image_size=768,
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        act_type='relu',
        backbone_args=dict(drop_path_rate=0.2),
        url='',
    ),
    tf_efficientdet_lite3=dict(
        name='tf_efficientdet_lite3',
        backbone_name='tf_efficientnet_lite3',
        image_size=896,
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        act_type='relu',
        backbone_args=dict(drop_path_rate=0.2),
        url='',
    ),
    tf_efficientdet_lite4=dict(
        name='tf_efficientdet_lite4',
        backbone_name='tf_efficientnet_lite4',
        image_size=1024,
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        act_type='relu',
        backbone_args=dict(drop_path_rate=0.2),
        url='',
    ),
)


def get_backbone_config(backbone_name='efficientnet_b0'):
    if backbone_name not in backbone_config:
        raise Exception("Backbone name {} not supported".format(backbone_name))
    return backbone_config[backbone_name]

def get_efficientdet_config(model_name='tf_efficientdet_d1'):
    """Get the default config for EfficientDet based on model name."""
    h = default_detection_model_configs()
    h.update(efficientdet_model_param_dict[model_name])
    return h


def bifpn_sum_config(base_reduction=8):
    """BiFPN config with sum."""
    p = OmegaConf.create()
    p.nodes = [
        {'reduction': base_reduction << 3, 'inputs_offsets': [3, 4]},
        {'reduction': base_reduction << 2, 'inputs_offsets': [2, 5]},
        {'reduction': base_reduction << 1, 'inputs_offsets': [1, 6]},
        {'reduction': base_reduction, 'inputs_offsets': [0, 7]},
        {'reduction': base_reduction << 1, 'inputs_offsets': [1, 7, 8]},
        {'reduction': base_reduction << 2, 'inputs_offsets': [2, 6, 9]},
        {'reduction': base_reduction << 3, 'inputs_offsets': [3, 5, 10]},
        {'reduction': base_reduction << 4, 'inputs_offsets': [4, 11]},
    ]
    p.weight_method = 'sum'
    return p


def bifpn_attn_config():
    """BiFPN config with fast weighted sum."""
    p = bifpn_sum_config()
    p.weight_method = 'attn'
    return p


def bifpn_fa_config():
    """BiFPN config with fast weighted sum."""
    p = bifpn_sum_config()
    p.weight_method = 'fastattn'
    return p


def get_fpn_config(fpn_name):
    if not fpn_name:
        fpn_name = 'bifpn_fa'
    name_to_config = {
        'bifpn_sum': bifpn_sum_config(),
        'bifpn_attn': bifpn_attn_config(),
        'bifpn_fa': bifpn_fa_config(),
    }
    return name_to_config[fpn_name]
