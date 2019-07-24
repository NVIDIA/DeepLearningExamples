Tensorflow BERT Samples
---

**Using bert_transformer Tensorflow op in a transformer encoder**

The trunk network of BERT model consists of a multi-layer transformer encoder,
which is implemented as the `transformer_model()` function in the file `modeling.py` in their official [Github repository](https://github.com/google-research/bert).
Samples provided in file `fast_infer_util.py` show how to re-implement this function with our ops in order to get an inference time speedup.


The function `fast_transformer_model_trans()` implements the transformer encoder using the `bert_transformer` op. 
In order to do that, we only need to first import the op at the beginning of the file, then call `bert_transformer` op at the end of each encoder layer. This turns out can be done by adding several lines of code to the original `transformer_model()` function as the following.

```python
# import op
transformer_op_module = tf.load_op_library(os.path.join('../../build/lib/libtf_fastertransformer.so'))
...
def fast_transformer_model_trans(...)
    ...
      # original code
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)

      # calling bert_transformer
      trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
      layer_output = transformer_op_module.bert_transformer(
        layer_input,
        layer_input,
        trainable_vars[0], trainable_vars[2], trainable_vars[4], trainable_vars[1], trainable_vars[3], trainable_vars[5], 
        attention_mask,
        trainable_vars[6], trainable_vars[7], trainable_vars[8], trainable_vars[9], trainable_vars[10], trainable_vars[11],
        trainable_vars[12], trainable_vars[13], trainable_vars[14], trainable_vars[15],
        batch_size=batch_size, from_seq_len=seq_length, to_seq_len=seq_length, head_num=num_attention_heads, size_per_head=attention_head_size)
      
      # original code
      prev_output = layer_output
      all_layer_outputs.append(layer_output)
    ...
```


**Running GLEU tasks with fast transformer inference**

The above shows how to implement a transformer encoder using our ops, to integrate it into the BERT pipeline
we can simply replace the `transformer_model` function in `modeling.py` with `fast_transformer_model_trans`.

Our implementation supports FP16 data type to further exploit the potential of inference acceleration.
FP16 inference was not supported by the original BERT code, here we made necessary modifications to build a FP16 compatible model,
which was implemented in `my_modeling.py` and the `create_model` function in `fast_infer_util.py`.

FP32 Tensorflow checkpoint files cannot be used directly for FP16 inference, we can convert its data type to FP16 in advance. 
The `ckpt_type_convert.py` script is provided for checkpoint data type conversion.

It is also important to note that our implementation requires a fixed batch size, this can be done by setting `drop_remainder` option to `True` for Tensorflow `Dataset` instances. We have re-implemented this as well in the `file_based_input_fn_builder_drop` function.

On top of the above modifications, it's easy to run any of the GLEU tasks supported by the open source BERT sample with our ops for better inference performance. We only need to replace several functions in original `run_classifier.py` script with the implementations we provide.

```python

import run_classifier as rc
import fast_infer_util as fiu
import my_modeling

...

# replace transformer implementation
my_modeling.transformer_model = fiu.fast_transformer_model_trans
# replace the model to support fp16 data type
rc.create_model = fiu.create_model
# replace the input function to drop remainder
rc.file_based_input_fn_builder = fiu.file_based_input_fn_builder_drop
...

```

The sample `run_classifier_wrap.py` is a wrapper of the original `run_classifier.py` script for BERT, it supports the same options as described in [BERT readme](https://github.com/google-research/bert) with additional `floatx` options to specify floating point type.
For example, to compare the performance of original BERT and our implementation on MRPC task we can run the following command.

```bash
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

python run_classifier.py   --task_name=MRPC   --do_eval=true   --data_dir=$GLUE_DIR/MRPC   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=ckpt_dir/fp32_model.ckpt   --max_seq_length=128   --eval_batch_size=8   --output_dir=mrpc_output

python run_classifier_wrap.py   --task_name=MRPC   --do_eval=true   --data_dir=$GLUE_DIR/MRPC   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=ckpt_dir/fp16_model.ckpt   --max_seq_length=128   --eval_batch_size=8   --output_dir=mrpc_output   --floatx=float16   

```

The evaluation result should be like

```
# original Tensorflow op
...
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.877451
INFO:tensorflow:  eval_loss = 0.44744828
INFO:tensorflow:  global_step = 0
INFO:tensorflow:  loss = 0.44744828

# faster_transformer op with fp16 data type
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.875
INFO:tensorflow:  eval_loss = 0.44731623
INFO:tensorflow:  global_step = 0
INFO:tensorflow:  loss = 0.44728807
...

```
We see the evaluation accuracy and loss drop slightly with FP16 inference for the MRPC sentence pair classification task.
The following section will show such minor sacrifice in accuracy will bring considerable performance gain.


**Tensorflow profiling**

The sample script `profile_transformer_inference.py` shows how to run and profile a BERT inference model from scratch. Results show we received a 6.36x speedup compared to FP32 Tensorflow (1.48x speedup compared to FP16 Tensorflow XLA) for an end-to-end classification model in our experiment settings.

GPU: Tesla T4  
CUDA: 10.0.0  
Model: BERT-Base: 12-layer, 768-hidden, 12-heads , 110M parameters  
Max sequence length: 128  
Batch size: 32  
Average time elapsed:

| settings  | seconds |
| ------------- | ------------- |
| FP32 Tensorflow | 0.2495 |
| FP32 Tensorflow XLA | 0.1998 |
| FP16 Tensorflow | 0.0978 |
| FP16 Tensorflow XLA | 0.0582 |
| FP16 FasterTransformer | 0.0392 |


**Content summary**

| file name  | summary |
| ------------- | ------------- |
| `ckpt_type_convert.py` | script for checkpoint data type conversion |
| `fast_infer_util.py` | example functions to use faster transformer ops in Tensorflow |
| `my_modeling.py` | basically the same as `modeling.py` in the original BERT repository, modifications are made to support FP16 data types |
| `run_classifier_wrap.py` | a wrapper script of `run_classifier.py` in the original BERT repository, shows how to run classification tasks using faster transformer ops |
| `profile_bert_inference.py` | for profiling BERT model pipelines |
| `profile_transformer_inference.py` | for profiling transformer encoder layers |
| `profile_util.py` | helper functions for profiling |
