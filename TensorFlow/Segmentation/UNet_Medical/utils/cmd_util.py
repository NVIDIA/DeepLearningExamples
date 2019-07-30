import argparse
import tensorflow as tf

PARSER = argparse.ArgumentParser(description="UNet-medical")

PARSER.add_argument('--exec_mode',
                    choices=['train', 'train_and_predict', 'predict'],
                    type=str,
                    default='train_and_predict',
                    help="""Which execution mode to run the model into"""
                    )

PARSER.add_argument('--model_dir',
                    type=str,
                    default='./results',
                    help="""Output directory for information related to the model"""
                    )

PARSER.add_argument('--data_dir',
                    type=str,
                    required=True,
                    help="""Input directory containing the dataset for training the model"""
                    )

PARSER.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help="""Size of each minibatch per GPU""")

PARSER.add_argument('--max_steps',
                    type=int,
                    default=1000,
                    help="""Maximum number of steps (batches) used for training""")

PARSER.add_argument('--seed',
                    type=int,
                    default=0,
                    help="""Random seed""")

PARSER.add_argument('--weight_decay',
                    type=float,
                    default=0.0005,
                    help="""Weight decay coefficient""")

PARSER.add_argument('--log_every',
                    type=int,
                    default=100,
                    help="""Log performance every n steps""")

PARSER.add_argument('--warmup_steps',
                    type=int,
                    default=200,
                    help="""Number of warmup steps""")

PARSER.add_argument('--learning_rate',
                    type=float,
                    default=0.01,
                    help="""Learning rate coefficient for SGD""")

PARSER.add_argument('--momentum',
                    type=float,
                    default=0.99,
                    help="""Momentum coefficient for SGD""")

PARSER.add_argument('--decay_steps',
                    type=float,
                    default=5000,
                    help="""Decay steps for inverse learning rate decay""")

PARSER.add_argument('--decay_rate',
                    type=float,
                    default=0.95,
                    help="""Decay rate for learning rate decay""")

PARSER.add_argument('--augment', dest='augment', action='store_true',
                    help="""Perform data augmentation during training""")
PARSER.add_argument('--no-augment', dest='augment', action='store_false')
PARSER.set_defaults(augment=False)

PARSER.add_argument('--benchmark', dest='benchmark', action='store_true',
                    help="""Collect performance metrics during training""")
PARSER.add_argument('--no-benchmark', dest='benchmark', action='store_false')
PARSER.set_defaults(augment=False)

PARSER.add_argument('--use_amp', dest='use_amp', action='store_true',
                    help="""Train using TF-AMP""")
PARSER.set_defaults(use_amp=False)

PARSER.add_argument('--use_trt', dest='use_trt', action='store_true',
                    help="""Use TF-TRT""")
PARSER.set_defaults(use_trt=False)


def _cmd_params(flags):
    return {
        'model_dir': flags.model_dir,
        'batch_size': flags.batch_size,
        'data_dir': flags.data_dir,
        'max_steps': flags.max_steps,
        'weight_decay': flags.weight_decay,
        'dtype': tf.float32,
        'learning_rate': flags.learning_rate,
        'momentum': flags.momentum,
        'benchmark': flags.benchmark,
        'augment': flags.augment,
        'exec_mode': flags.exec_mode,
        'seed': flags.seed,
        'use_amp': flags.use_amp,
        'use_trt': flags.use_trt,
        'log_every': flags.log_every,
        'warmup_steps': flags.warmup_steps,
        'decay_steps': flags.decay_steps,
        'decay_rate': flags.decay_rate,
    }
