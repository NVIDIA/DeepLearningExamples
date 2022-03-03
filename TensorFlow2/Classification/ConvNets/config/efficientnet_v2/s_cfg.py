import tensorflow as tf

# NOTE: this confile file can further be overridden by user-defined params provided at the command line



config = dict(
        path_to_impl='model.efficientnet_model_v2',
        
        #data-related model params
        num_classes=1000,  # must be the same as data.num_classes
        input_channels= 3,
        rescale_input= 1, # binary
        mean_rgb=(0.485 * 255, 0.456 * 255, 0.406 * 255), # used when rescale_input=True
        std_rgb=(0.229 * 255, 0.224 * 255, 0.225 * 255), # used when rescale_input=True
        dtype= tf.float32, #used for input image normalization/casting,  # tf.float32, tf.bfloat16,  tf.float16,  tf.float32,  tf.bfloat16,
        
        # GUIDE
        #                                       width   depth  resolution dropout
        #      efficientnet_v2-s               1.0      1.0      300        0.2

        width_coefficient= 1.0,
        depth_coefficient= 1.0,
        dropout_rate= 0.2, # used in the cls head
        # image resolution must be set in tr/eval/predict configs below
        
        drop_connect_rate= 0.2, # used in residual for stochastic depth
        conv_dropout= None, # used in pre-SE, but never used
        stem_base_filters= 24, # effnetv2
        top_base_filters= 1280,
        activation= 'swish', # same as silu
        depth_divisor= 8,
        min_depth=8, 
        # use_se= True, # No longer global: blocks may or may not have it.
        batch_norm= 'syncbn',
        bn_momentum= 0.99, # google uses 0.9
        bn_epsilon= 1e-3,
        weight_init= 'fan_out', # google uses untruncated

        # NEW
        # gn_groups=8, # group normalization
        # local_pooling=0, # as opposed global pooling for SE
        # headbias=None, # bias for cls head

        blocks= (
            # (input_filters, output_filters, kernel_size, num_repeat,expand_ratio, strides, se_ratio)
            # pylint: disable=bad-whitespace
            dict(input_filters=24,   output_filters=24,   kernel_size=3, num_repeat=2, expand_ratio=1, strides=(1, 1), se_ratio=None,id_skip=True,fused_conv=True,conv_type=None),
            dict(input_filters=24,   output_filters=48,   kernel_size=3, num_repeat=4, expand_ratio=4, strides=(2, 2), se_ratio=None,id_skip=True,fused_conv=True,conv_type=None),
            dict(input_filters=48,   output_filters=64,   kernel_size=3, num_repeat=4, expand_ratio=4, strides=(2, 2), se_ratio=None,id_skip=True,fused_conv=True,conv_type=None),
            dict(input_filters=64,   output_filters=128,   kernel_size=3, num_repeat=6, expand_ratio=4, strides=(2, 2), se_ratio=0.25,id_skip=True,fused_conv=False,conv_type='depthwise'),
            dict(input_filters=128,   output_filters=160,  kernel_size=3, num_repeat=9, expand_ratio=6, strides=(1, 1), se_ratio=0.25,id_skip=True,fused_conv=False,conv_type='depthwise'),
            dict(input_filters=160,  output_filters=256,  kernel_size=3, num_repeat=15, expand_ratio=6, strides=(2, 2), se_ratio=0.25,id_skip=True,fused_conv=False,conv_type='depthwise'),
            # pylint: enable=bad-whitespace
        ),
    )