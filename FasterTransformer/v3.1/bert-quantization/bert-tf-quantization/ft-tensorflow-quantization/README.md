# FasterTransformer TensorFlow Quantization

The purpose of this repo is to build minimum quantization functionality to quantize BERT.

For now, this tool has:
  - fake_quantize function
  - FakeQuantizer (TensorQuantizer)
  - QuantDense
  - Max and Histogram Collector
  - Calibrator which supports Max, Percentile, MSE calibration methods

## Install

```bash
pip install .
```

## Usage

### Quantize a model
To convert a model to a quantized version for post training quantization or quantization-aware finetuning, it can be easily done by adding tensor quantizer nodes and replacing dense nodes.
Here is an example:

```python
import tensorflow as tf
import tensorflow.layers.Dense as Dense

from ft_tensorflow_quantization import FakeQuantizer, QuantDense

def simple_model(x, y):
  dense1 = Dense(128, name='dense1')
  dense2 = Dense(128, name='dense2')
  output = tf.matmul(dense1(x), dense2(y), transpose_b=True)
  return output

def quant_simple_model(x, y, if_quant):
  dense1 = QuantDense(128, name='dense1', if_quant=if_quant)
  dense2 = QuantDense(128, name='dense2', if_quant=if_quant)

  input_desc = QuantDense.default_quant_desc_input
  dense1_output_quantizer = FakeQuantizer(input_desc, if_quant)
  dense2_output_quantizer = FakeQuantizer(input_desc, if_quant)
  dense1_output = dense1_output_quantizer(dense1(x))
  dense2_output = dense2_output_quantizer(dense2(x))
  output = tf.matmul(dense1_output, dense2_output, transpose_b=True)
  return output
```

### Calibration
Here are the steps to do calibration on some data after building the model:

```python
calibrator_lists = {}
with tf.compat.v1.Session() as sess:
  sess.run(tf.compat.v1.global_variables_initializer())

  # get calibrators
  calibrator_lists['input'] = get_calibrators('input', collector_type='histogram')
  calibrator_lists['kernel'] = get_calibrators('kernel', collector_type='max', axis=1)

  # collect
  step_op_list = []
  for _, calib_list in calibrator_lists.items():
    for calibrator in calib_list:
      step_op_list.append(calibrator.calib_step_op(sess.graph))
  for i in range(step):
    sess.run(step_op_list, feed_dict)

  # compute range
  for calibrator in self.calibrator_lists['input']:
    calibrator.compute_range('percentile', percentile=99.99)
  for calibrator in self.calibrator_lists['kernel']:
    calibrator.compute_range('max')

  # load range back to the model
  placeholders = {}
  load_min_op = {}
  load_max_op = {}
  for _, calib_list in calibrator_lists.items():
    for calibrator in calib_list:
      placeholders[calibrator.tensor_name_prefix] = tf.placeholder(tf.float32)
      load_min_op[calibrator.tensor_name_prefix] = tf.compat.v1.assign(
          graph.get_tensor_by_name(calibrator.quant_min_name), placeholders[calibrator.tensor_name_prefix])
      load_max_op[calibrator._tensor_name_prefix] = tf.compat.v1.assign(
          graph.get_tensor_by_name(calibrator.quant_max_name), placeholders[calibrator.tensor_name_prefix])
  sess.run(load_min_op,
      {placeholders[calibrator.tensor_name_prefix]:calibrator.calib_min \
          for _, calib_list in calibrator_lists.items() for calibrator in calib_list})
  sess.run(load_max_op,
      {placeholders[calibrator.tensor_name_prefix]:calibrator.calib_max \
          for _, calib_list in calibrator_lists.items() for calibrator in calib_list})
```

### Calibration Hook
If the training and evaluation are implemented by estimator of TensorFlow, a calibration hook is needed for quantization.
Here is a sample hook code:

```python
class CalibrationHook(tf.train.SessionRunHook):
  def __init__(self):
    self.calibrator_lists = {}

  def begin(self):
    self.saver = tf.train.Saver()
    tf.compat.v1.logging.info("initializing calibrators")
    graph = tf.compat.v1.get_default_graph()
    self.calibrator_lists['input'] = get_calibrators('input', collector_type='histogram')
    self.calibrator_lists['kernel'] = get_calibrators('kernel', collector_type='max', axis=1)
    for k in ['input', 'kernel']:
      tf.compat.v1.logging.info("There are {} calibrators in collection '{}'".format(len(self.calibrator_lists[k]), k))

    self.calib_step = [
      calibrator.calib_step_op(graph) for _, calib_list in self.calibrator_lists.items() for calibrator in calib_list]

    self.placeholders = {}
    self.load_min_op = {}
    self.load_max_op = {}
    for _, calib_list in self.calibrator_lists.items():
      for calibrator in calib_list:
        if calibrator.tensor_name_prefix in self.placeholders:
          raise ValueError("repeated name prefix")
        self.placeholders[calibrator.tensor_name_prefix] = tf.placeholder(tf.float32)
        self.load_min_op[calibrator.tensor_name_prefix] = tf.compat.v1.assign(
          graph.get_tensor_by_name(calibrator.quant_min_name), self.placeholders[calibrator.tensor_name_prefix])
        self.load_max_op[calibrator._tensor_name_prefix] = tf.compat.v1.assign(
          graph.get_tensor_by_name(calibrator.quant_max_name), self.placeholders[calibrator.tensor_name_prefix])

  def before_run(self, run_context):
    tf.compat.v1.logging.info("registering calibration step")
    return tf.estimator.SessionRunArgs(fetches=self.calib_step)

  def end(self, session):
    tf.compat.v1.logging.info("computing calibration ranges")
    if FLAGS.calib_method == 'percentile':
      tf.compat.v1.logging.info("percentile calibration with value {}.".format(FLAGS.percentile))
      for calibrator in self.calibrator_lists['input']:
          calibrator.compute_range('percentile', percentile=FLAGS.percentile)
    elif FLAGS.calib_method == 'mse':
      tf.compat.v1.logging.info("mse calibration.")
      for calibrator in self.calibrator_lists['input']:
          calibrator.compute_range('mse')
    else:
      raise ValueError("Unsupported calibration method.")
    for calibrator in self.calibrator_lists['kernel']:
      calibrator.compute_range('max')

    tf.compat.v1.logging.info("loading calibration ranges")
    session.run(self.load_min_op,
      {self.placeholders[calibrator.tensor_name_prefix]:calibrator.calib_min \
        for _, calib_list in self.calibrator_lists.items() for calibrator in calib_list})
    session.run(self.load_max_op,
      {self.placeholders[calibrator.tensor_name_prefix]:calibrator.calib_max \
        for _, calib_list in self.calibrator_lists.items() for calibrator in calib_list})
    tf.compat.v1.logging.info("saving calibrated model")
    with open(FLAGS.calibrator_file, 'wb') as f:
      pickle.dump(self.calibrator_lists, f)
    self.saver.save(session, os.path.join(FLAGS.output_dir, 'model.ckpt-calibrated'))
```

### QAT
For quantization aware training, it is the same as the usual finetuning, but starting from a quantized model and calibrated checkpoint.
