import argparse
import os
import pathlib
import time
import tempfile

import tensorflow as tf
import numpy as np

from tensorflow.python.compiler.tensorrt import trt_convert as trt

import dllogger

from runtime import runner_utils
from runtime import runner
from model.resnet import model_architectures
from utils import data_utils
from utils import hvd_wrapper as hvd

OUTPUT_SAVED_MODEL_PATH = tempfile.mkdtemp(prefix="tftrt-converted")
LOG_FREQUENCY = 100

def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    exclusive_args = parser.add_mutually_exclusive_group()
    exclusive_args.add_argument("--model", type=str, default=None, help="Saved model location to use for inference")
    exclusive_args.add_argument("--architecture", type=str, choices=model_architectures.keys())

    parser.add_argument("--log-path", type=str, default="./log.json", help="Path to log file")
    parser.add_argument("--tf-trt", action="store_true", default=False, help="Use TF-TRT for inference")
    parser.add_argument("--amp", action="store_true", default=False, help="Use AMP for inference")
    parser.add_argument("--data-dir", type=str, required=False, 
                        default=None, help="Localization of validation data")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")

    return parser.parse_args()

def main(args: argparse.Namespace):
    hvd.init()

    dllogger.init(backends=[
        dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE, filename=args.log_path),
        dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)
    ])
    dllogger.log(data=vars(args), step='PARAMETER')
    dllogger.metadata("throughput", {"unit": "images/s"})
    dllogger.metadata("accuracy", {"unit": None})

    if args.model is None:
        saved_model_to_load = tempfile.mkdtemp(prefix="tftrt-savedmodel")
        r = runner.Runner(n_classes=1001, architecture=args.architecture, use_tf_amp=args.amp, 
            model_dir=saved_model_to_load)
        r.train("batch", 1, 1, args.batch_size, is_benchmark=True)
        r.evaluate("batch", 1, args.batch_size, export_dir=saved_model_to_load, 
            is_benchmark=True)

        saved_model_to_load = r.exported_path.decode("utf-8")
    else:
        saved_model_to_load = args.model

    output_tensor_name = "y_preds_ref:0" if not args.tf_trt else "ArgMax:0"
    batch_size = args.batch_size

    if args.tf_trt:
        converter = trt.TrtGraphConverter(input_saved_model_dir=str(saved_model_to_load),
                                          precision_mode="FP16" if args.amp else "FP32")
        converter.convert()
        converter.save(OUTPUT_SAVED_MODEL_PATH)
        saved_model_to_load = OUTPUT_SAVED_MODEL_PATH
    elif args.amp:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

    if args.data_dir is not None:
        filenames, _, num_steps, _, _ = runner_utils.parse_tfrecords_dataset(
                    data_dir=str(args.data_dir),
                    mode="validation",
                    iter_unit="epoch",
                    num_iter=1,
                    global_batch_size=batch_size,
                )


        dataset = data_utils.get_tfrecords_input_fn(filenames=filenames,
                                                    batch_size=batch_size,
                                                    height=224,
                                                    width=224,
                                                    training=False,
                                                    distort_color=False,
                                                    num_threads=1,
                                                    deterministic=True)
        iterator = dataset.make_initializable_iterator()
        next_item = iterator.get_next()
    else:
        num_steps=60000 / batch_size
    

    with tf.Session() as sess:
        if args.data_dir is not None:
            sess.run(iterator.initializer)
        tf.saved_model.loader.load(sess, 
            [tf.saved_model.tag_constants.SERVING],
            str(saved_model_to_load))

        try:
            start_time = time.time()
            last_time = start_time
            image_processed = 0
            image_correct = 0

            for samples_processed in range(int(num_steps)):
                if args.data_dir is not None:
                    next_batch_image, next_batch_target = sess.run(next_item)
                else:
                    if samples_processed == 0:
                        next_batch_image = np.random.normal(size=(batch_size, 224, 224, 3))
                        next_batch_target = np.random.randint(0, 1000, size=(batch_size,))
                output = sess.run([output_tensor_name], feed_dict={"input_tensor:0": next_batch_image})
                image_processed += args.batch_size
                image_correct += np.sum(output == next_batch_target)

                if samples_processed % LOG_FREQUENCY == 0 and samples_processed != 0:
                    current_time = time.time()
                    current_throughput = LOG_FREQUENCY * batch_size / (current_time - last_time)
                    dllogger.log(step=(0, samples_processed), data={"throughput": current_throughput})
                    last_time = current_time

        except tf.errors.OutOfRangeError:
            pass
        finally:
            dllogger.log(step=tuple(), data={"throughput": image_processed / (last_time - start_time), 
                                             "accuracy": image_correct / image_processed})


if __name__ == "__main__":
    main(argument_parser())