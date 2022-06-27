# Copyright 2020 Google Research. All Rights Reserved.
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
"""Tool to inspect a model."""
import os

from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow as tf
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
import dllogger as DLLogger

from model import inference
from utils import hparams_config
from utils import model_utils
from utils import setup

flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model.')
flags.DEFINE_string('mode', 'benchmark',
                    'Run mode: {dry, export, benchmark}')
flags.DEFINE_string('trace_filename', None, 'Trace file name.')

flags.DEFINE_integer('bm_runs', 100, 'Number of benchmark runs.')
flags.DEFINE_string('tensorrt', None, 'TensorRT mode: {None, FP32, FP16, INT8}')
flags.DEFINE_integer('batch_size', 1, 'Batch size for inference.')

flags.DEFINE_string('ckpt_path', '_', 'checkpoint dir used for eval.')
flags.DEFINE_string('export_ckpt', None, 'Output model ckpt path.')

flags.DEFINE_string(
    'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
    ' containing attributes to use as hyperparameters.')

flags.DEFINE_bool('amp', True, 'Enable mixed precision training')
flags.DEFINE_bool('use_xla', True, 'Use XLA')
flags.DEFINE_string('input_image', None, 'Input image path for inference.')
flags.DEFINE_string('output_image_dir', None, 'Output dir for inference.')
flags.DEFINE_string('dllogger_path', '/tmp/time_log.txt', 'Filepath for dllogger logs')

# For video.
flags.DEFINE_string('input_video', None, 'Input video path for inference.')
flags.DEFINE_string('output_video', None,
                    'Output video path. If None, play it online instead.')

# For visualization.
flags.DEFINE_integer('max_boxes_to_draw', 100, 'Max number of boxes to draw.')
flags.DEFINE_float('min_score_thresh', 0.4, 'Score threshold to show box.')
flags.DEFINE_string('nms_method', 'hard', 'nms method, hard or gaussian.')

# For saved model.
flags.DEFINE_string('saved_model_dir', None,
                    'Folder path for saved model.')
flags.DEFINE_string('tflite_path', None, 'Path for exporting tflite file.')
flags.DEFINE_bool('debug', False, 'Debug mode.')
FLAGS = flags.FLAGS


def main(_):
  model_config = hparams_config.get_detection_config(FLAGS.model_name)
  model_config.override(FLAGS.hparams)  # Add custom overrides
  model_config.is_training_bn = False
  model_config.image_size = model_utils.parse_image_size(model_config.image_size)

  # A hack to make flag consistent with nms configs.
  if FLAGS.min_score_thresh:
    model_config.nms_configs.score_thresh = FLAGS.min_score_thresh
  if FLAGS.nms_method:
    model_config.nms_configs.method = FLAGS.nms_method
  if FLAGS.max_boxes_to_draw:
    model_config.nms_configs.max_output_size = FLAGS.max_boxes_to_draw
  model_config.mixed_precision = FLAGS.amp

  setup.set_flags(FLAGS, model_config, training=False)
  model_params = model_config.as_dict()
  ckpt_path_or_file = FLAGS.ckpt_path
  if tf.io.gfile.isdir(ckpt_path_or_file):
    ckpt_path_or_file = tf.train.latest_checkpoint(ckpt_path_or_file)
  driver = inference.ServingDriver(FLAGS.model_name, ckpt_path_or_file,
                                   FLAGS.batch_size or None,
                                   FLAGS.min_score_thresh,
                                   FLAGS.max_boxes_to_draw, model_params)
  # dllogger setup
  backends = []
  backends+=[
    JSONStreamBackend(verbosity=Verbosity.VERBOSE, filename=FLAGS.dllogger_path),
    StdOutBackend(verbosity=Verbosity.DEFAULT)]
  DLLogger.init(backends=backends)
  DLLogger.metadata('inference_fps', {'unit': 'images/s'})
  DLLogger.metadata('inference_latency_ms', {'unit': 'ms'})
  DLLogger.metadata('latency_avg', {'unit': 's'})
  DLLogger.metadata('latency_90', {'unit': 's'})
  DLLogger.metadata('latency_95', {'unit': 's'})
  DLLogger.metadata('latency_99', {'unit': 's'})

  if FLAGS.mode == 'export':
    if tf.io.gfile.exists(FLAGS.saved_model_dir):
      tf.io.gfile.rmtree(FLAGS.saved_model_dir)
    driver.export(FLAGS.saved_model_dir, FLAGS.tflite_path, FLAGS.tensorrt)
  elif FLAGS.mode == 'benchmark':
    if FLAGS.saved_model_dir:
      driver.load(FLAGS.saved_model_dir)

    batch_size = FLAGS.batch_size or 1
    if FLAGS.input_image:
      image_file = tf.io.read_file(FLAGS.input_image)
      image_arrays = tf.image.decode_image(image_file)
      image_arrays.set_shape((None, None, 3))
      image_arrays = tf.expand_dims(image_arrays, 0)
      if batch_size > 1:
        image_arrays = tf.tile(image_arrays, [batch_size, 1, 1, 1])
    else:
      # use synthetic data if no image is provided.
      image_arrays = tf.ones((batch_size, *model_config.image_size, 3),
                             dtype=tf.uint8)
    driver.benchmark(image_arrays, FLAGS.bm_runs, FLAGS.trace_filename)
  elif FLAGS.mode == 'dry':
    # transfer to tf2 format ckpt
    driver.build()
    if FLAGS.export_ckpt:
      driver.model.save_weights(FLAGS.export_ckpt)
  elif FLAGS.mode == 'video':
    import cv2  # pylint: disable=g-import-not-at-top
    if tf.saved_model.contains_saved_model(FLAGS.saved_model_dir):
      driver.load(FLAGS.saved_model_dir)
    cap = cv2.VideoCapture(FLAGS.input_video)
    if not cap.isOpened():
      print('Error opening input video: {}'.format(FLAGS.input_video))

    out_ptr = None
    if FLAGS.output_video:
      frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
      out_ptr = cv2.VideoWriter(FLAGS.output_video,
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25,
                                (frame_width, frame_height))

    while cap.isOpened():
      # Capture frame-by-frame
      ret, frame = cap.read()
      if not ret:
        break

      raw_frames = np.array([frame])
      detections_bs = driver.serve(raw_frames)
      boxes, scores, classes, _ = tf.nest.map_structure(np.array, detections_bs)
      new_frame = driver.visualize(
          raw_frames[0],
          boxes[0],
          scores[0],
          classes[0],
          min_score_thresh=model_config.nms_configs.score_thresh,
          max_boxes_to_draw=model_config.nms_configs.max_output_size)

      if out_ptr:
        # write frame into output file.
        out_ptr.write(new_frame)
      else:
        # show the frame online, mainly used for real-time speed test.
        cv2.imshow('Frame', new_frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  app.run(main)
