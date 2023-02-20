"""
Script to benchmark model throughput and latency
"""
import os
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
import hydra
from omegaconf import DictConfig
import tensorflow as tf
from tensorflow.keras import mixed_precision

from data_generators import tf_data_generator
from utils.general_utils import join_paths, suppress_warnings
from utils.images_utils import postprocess_mask
from models.model import prepare_model


def benchmark_time(cfg: DictConfig):
    """
    Output throughput and latency
    """

    # suppress TensorFlow and DALI warnings
    suppress_warnings()

    if cfg.OPTIMIZATION.AMP:
        print("Enabling Automatic Mixed Precision(AMP)")
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    if cfg.OPTIMIZATION.XLA:
        print("Enabling Accelerated Linear Algebra(XLA)")
        tf.config.optimizer.set_jit(True)

    # data generator
    val_generator = tf_data_generator.DataGenerator(cfg, mode="VAL")
    validation_steps = val_generator.__len__()

    warmup_steps, bench_steps = 50, 100
    if "warmup_steps" in cfg.keys():
        warmup_steps = cfg.warmup_steps
    if "bench_steps" in cfg.keys():
        bench_steps = cfg.bench_steps
    validation_steps = min(validation_steps, (warmup_steps + bench_steps))

    progress_bar = tqdm(total=validation_steps)

    # create model
    model = prepare_model(cfg)

    # weights model path
    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH,
        f"{cfg.MODEL.WEIGHTS_FILE_NAME}.hdf5"
    )

    assert os.path.exists(checkpoint_path), \
        f"Model weight's file does not exist at \n{checkpoint_path}"

    # load model weights
    model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
    # model.summary()

    time_taken = []
    # for each batch
    for i, (batch_images, batch_mask) in enumerate(val_generator):

        start_time = timer()
        # make prediction on batch
        batch_predictions = model.predict_on_batch(batch_images)
        if len(model.outputs) > 1:
            batch_predictions = batch_predictions[0]

        # do postprocessing on predicted mask
        batch_predictions = postprocess_mask(batch_predictions, cfg.OUTPUT.CLASSES)

        time_taken.append(timer() - start_time)

        progress_bar.update(1)
        if i >= validation_steps:
            break
    progress_bar.close()

    mean_time = np.mean(time_taken[warmup_steps:])  # skipping warmup_steps
    throughput = (cfg.HYPER_PARAMETERS.BATCH_SIZE / mean_time)
    print(f"Latency: {round(mean_time * 1e3, 2)} msec")
    print(f"Throughput/FPS: {round(throughput, 2)} samples/sec")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Read config file and pass to benchmark_time method
    """
    benchmark_time(cfg)


if __name__ == "__main__":
    main()
