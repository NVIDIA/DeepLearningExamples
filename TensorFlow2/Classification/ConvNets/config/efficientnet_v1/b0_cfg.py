import tensorflow as tf
from config.defaults import Config

# NOTE: this confile file can further be overridden by user-defined params provided at the command line


config = dict(

        path_to_impl='model.efficientnet_model_v1',


        
        #data-related model params
        num_classes=1000,  # must be the same as data.num_classes
        input_channels= 3,
        rescale_input= 1, # binary,
        mean_rgb=(0.485 * 255, 0.456 * 255, 0.406 * 255), # used when rescale_input=True
        std_rgb=(0.229 * 255, 0.224 * 255, 0.225 * 255), # used when rescale_input=True
        dtype= tf.float32, #used for input image normalization/casting,  # tf.float32, tf.bfloat16,  tf.float16,  tf.float32,  tf.bfloat16,
        
        
        # GUIDE
        #                                       width   depth  resolution dropout
        #      efficientnet_v1-b0               1.0      1.0      224        0.2
        #     'efficientnet_v1-b1               1.0      1.1      240        0.2
        #     'efficientnet_v1-b2               1.1      1.2      260        0.3
        #     'efficientnet_v1-b3               1.2      1.4      300        0.3
        #     'efficientnet_v1-b4               1.4      1.8      380        0.4
        #     'efficientnet_v1-b5               1.6      2.2      456        0.4
        #     'efficientnet_v1-b6               1.8      2.6      528        0.5
        #     'efficientnet_v1-b7               2.0      3.1      600        0.5
        #     'efficientnet_v1-b8               2.2      3.6      672        0.5 
        #     'efficientnet_v1-l2               4.3      5.3      800        0.5
        width_coefficient= 1.0,
        depth_coefficient= 1.0,
        dropout_rate= 0.2,
        # image resolution must be set in tr/eval/predict configs below
                
        drop_connect_rate= 0.2,
        stem_base_filters= 32,
        top_base_filters= 1280,
        activation= 'swish',
        depth_divisor= 8,
        min_depth= None,
        use_se= 1, # binary
        batch_norm= 'syncbn',
        bn_momentum= 0.99,
        bn_epsilon= 1e-3,
        weight_init= 'fan_out',
        
        blocks= (
            # (input_filters, output_filters, kernel_size, num_repeat,expand_ratio, strides, se_ratio)
            # pylint: disable=bad-whitespace
            dict(input_filters=32,   output_filters=16,   kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25,id_skip=True,fused_conv=False,conv_type='depthwise'),
            dict(input_filters=16,   output_filters=24,   kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25,id_skip=True,fused_conv=False,conv_type='depthwise'),
            dict(input_filters=24,   output_filters=40,   kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25,id_skip=True,fused_conv=False,conv_type='depthwise'),
            dict(input_filters=40,   output_filters=80,   kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25,id_skip=True,fused_conv=False,conv_type='depthwise'),
            dict(input_filters=80,   output_filters=112,  kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25,id_skip=True,fused_conv=False,conv_type='depthwise'),
            dict(input_filters=112,  output_filters=192,  kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25,id_skip=True,fused_conv=False,conv_type='depthwise'),
            dict(input_filters=192,  output_filters=320,  kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25,id_skip=True,fused_conv=False,conv_type='depthwise'),
            # pylint: enable=bad-whitespace
        ),
 
    )

# train_config = dict(lr_decay='cosine',
#
#                     max_epochs=500,
#                     img_size=224,
#                     batch_size=256,
#                     save_checkpoint_freq=5,
#                     lr_init=0.005,
#                     weight_decay=5e-6,
#                     epsilon=0.001,
#                     resume_checkpoint=1,
#                     enable_tensorboard=0
#                    )
#
# eval_config = dict(img_size=224,
#                    batch_size=256)
#
# data_config = dict(
#     data_dir='/data/',
#     augmenter_name='autoaugment',
#     mixup_alpha=0.0,
#
#
# )
# runtime_config = dict(mode='train_and_eval',
#                       model_dir='./output/',
#                       use_amp=1,
#                       use_xla=1,
#                       log_steps=100
#                       )
#
# config = dict(model=model_config,
#               train=train_config,
#               eval=eval_config,
#               data=data_config,
#               runtime=runtime_config,
#               )
