import argparse

import tensorflow as tf

from dlexport.tensorflow import to_savedmodel, to_onnx, to_tensorrt
from utils.data_loader import Dataset
from utils.model_fn import unet_fn

PARSER = argparse.ArgumentParser(description="U-Net medical")

PARSER.add_argument('--to', dest='to', choices=['savedmodel', 'tensorrt', 'onnx'], required=True)

PARSER.add_argument('--use_amp', dest='use_amp', action='store_true', default=False)
PARSER.add_argument('--use_xla', dest='use_xla', action='store_true', default=False)
PARSER.add_argument('--compress', dest='compress', action='store_true', default=False)

PARSER.add_argument('--input_shape',
                    nargs='+',
                    type=int,
                    help="""Directory where to download the dataset""")

PARSER.add_argument('--data_dir',
                    type=str,
                    help="""Directory where to download the dataset""")

PARSER.add_argument('--checkpoint_dir',
                    type=str,
                    help="""Directory where to download the dataset""")

PARSER.add_argument('--savedmodel_dir',
                    type=str,
                    help="""Directory where to download the dataset""")

PARSER.add_argument('--precision',
                    type=str,
                    choices=['FP32', 'FP16', 'INT8'],
                    help="""Directory where to download the dataset""")


def main():
    """
    Starting point of the application
    """
    flags = PARSER.parse_args()

    if flags.to == 'savedmodel':
        to_savedmodel(input_shape=flags.input_shape,
                      model_fn=unet_fn,
                      src_dir=flags.checkpoint_dir,
                      dst_dir='./saved_model',
                      input_names=['IteratorGetNext'],
                      output_names=['total_loss_ref'],
                      use_amp=flags.use_amp,
                      use_xla=flags.use_xla,
                      compress=flags.compress)
    if flags.to == 'tensorrt':
        ds = Dataset(data_dir=flags.data_dir,
                     batch_size=1,
                     augment=False,
                     gpu_id=0,
                     num_gpus=1,
                     seed=42)
        iterator = ds.test_fn(count=1).make_one_shot_iterator()
        features = iterator.get_next()

        sess = tf.Session()

        def input_data():
            return {'input_tensor:0': sess.run(features)}

        to_tensorrt(src_dir=flags.savedmodel_dir,
                  dst_dir='./tf_trt_model',
                  precision=flags.precision,
                  feed_dict_fn=input_data,
                  num_runs=1,
                  output_tensor_names=['Softmax:0'],
                  compress=flags.compress)
    if flags.to == 'onnx':
        to_onnx(src_dir=flags.savedmodel_dir,
                dst_dir='./onnx_model',
                compress=flags.compress)


if __name__ == '__main__':
    main()

