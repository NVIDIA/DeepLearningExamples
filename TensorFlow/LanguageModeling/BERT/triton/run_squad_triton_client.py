# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import modeling
import tokenization
from tensorrtserver.api import ProtocolType, InferContext, ServerStatusContext, grpc_service_pb2_grpc, grpc_service_pb2, model_config_pb2
from utils.create_squad_data import *
import grpc
from run_squad import write_predictions, get_predictions, RawResult
import numpy as np
import tqdm
from functools import partial

import sys
if sys.version_info >= (3, 0):
  import queue
else:
  import Queue as queue


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

# Triton Specific flags
flags.DEFINE_string("triton_model_name", "bert", "exports to appropriate directory for Triton")
flags.DEFINE_integer("triton_model_version", 1, "exports to appropriate directory for Triton")
flags.DEFINE_string("triton_server_url", "localhost:8001", "exports to appropriate directory for Triton")

# Input Text for Inference
flags.DEFINE_string("question", None, "Question for Inference")
flags.DEFINE_string("context", None, "Context for Inference")
flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")


# Set this to either 'label_ids' for Google bert or 'unique_ids' for JoC
label_id_key = "unique_ids"

# User defined class to store infer_ctx and request id
# from callback function and let main thread to handle them
class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

# Callback function used for async_run(), it can capture
# additional information using functools.partial as long as the last
# two arguments are reserved for InferContext and request id
def completion_callback(user_data, idx, start_time, inputs, infer_ctx, request_id):
    user_data._completed_requests.put((infer_ctx, request_id, idx, start_time, inputs))

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        label_ids_data = ()
        input_ids_data = ()
        input_mask_data = ()
        segment_ids_data = ()
        for i in range(0, min(n, l-ndx)):
            label_ids_data = label_ids_data + (np.array([iterable[ndx + i].unique_id], dtype=np.int32),)
            input_ids_data = input_ids_data+ (np.array(iterable[ndx + i].input_ids, dtype=np.int32),)
            input_mask_data = input_mask_data+ (np.array(iterable[ndx + i].input_mask, dtype=np.int32),)
            segment_ids_data = segment_ids_data+ (np.array(iterable[ndx + i].segment_ids, dtype=np.int32),)

        inputs_dict = {label_id_key: label_ids_data,
                       'input_ids': input_ids_data,
                       'input_mask': input_mask_data,
                       'segment_ids': segment_ids_data}
        yield inputs_dict

def main(_):
    """
    Ask a question of context on Triton.
    :param context: str
    :param question: str
    :param question_id: int
    :return:
    """
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_lazy_compilation=false" #causes memory fragmentation for bert leading to OOM

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # Get the Data
    if FLAGS.predict_file:
        eval_examples = read_squad_examples(
            input_file=FLAGS.predict_file, is_training=False,
            version_2_with_negative=FLAGS.version_2_with_negative)
    elif FLAGS.question and FLAGS.answer:
        input_data = [{"paragraphs":[{"context":FLAGS.context,
                        "qas":[{"id":0, "question":FLAGS.question}]}]}]

        eval_examples = read_squad_examples(input_file=None, is_training=False,
            version_2_with_negative=FLAGS.version_2_with_negative, input_data=input_data)
    else:
        raise ValueError("Either predict_file or question+answer need to defined")
    
    # Get Eval Features = Preprocessing
    eval_features = []
    def append_feature(feature):
        eval_features.append(feature)

    convert_examples_to_features(
        examples=eval_examples[0:],
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)

    protocol_str = 'grpc' # http or grpc
    url = FLAGS.triton_server_url
    verbose = True
    model_name = FLAGS.triton_model_name
    model_version = FLAGS.triton_model_version
    batch_size = FLAGS.predict_batch_size

    protocol = ProtocolType.from_str(protocol_str) # or 'grpc'

    ctx = InferContext(url, protocol, model_name, model_version, verbose)

    status_ctx = ServerStatusContext(url, protocol, model_name=model_name, verbose=verbose)

    model_config_pb2.ModelConfig()

    status_result = status_ctx.get_server_status()
    user_data = UserData()

    max_outstanding = 20
    # Number of outstanding requests
    outstanding = 0

    sent_prog = tqdm.tqdm(desc="Send Requests", total=len(eval_features))
    recv_prog = tqdm.tqdm(desc="Recv Requests", total=len(eval_features))

    def process_outstanding(do_wait, outstanding):

        if (outstanding == 0 or do_wait is False):
            return outstanding

        # Wait for deferred items from callback functions
        (infer_ctx, ready_id, idx, start_time, inputs) = user_data._completed_requests.get()

        if (ready_id is None):
            return outstanding

        # If we are here, we got an id
        result = ctx.get_async_run_results(ready_id)
        stop = time.time()

        if (result is None):
            raise ValueError("Context returned null for async id marked as done")

        outstanding -= 1

        time_list.append(stop - start_time)

        batch_count = len(inputs[label_id_key])

        for i in range(batch_count):
            unique_id = int(inputs[label_id_key][i][0])
            start_logits = [float(x) for x in result["start_logits"][i].flat]
            end_logits = [float(x) for x in result["end_logits"][i].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

        recv_prog.update(n=batch_count)
       	return outstanding

    all_results = []
    time_list = []

    print("Starting Sending Requests....\n")

    all_results_start = time.time()
    idx = 0
    for inputs_dict in batch(eval_features, batch_size):

        present_batch_size = len(inputs_dict[label_id_key])

        outputs_dict = {'start_logits': InferContext.ResultFormat.RAW,
                        'end_logits': InferContext.ResultFormat.RAW}

        start_time = time.time()
        ctx.async_run(partial(completion_callback, user_data, idx, start_time, inputs_dict),
        	inputs_dict, outputs_dict, batch_size=present_batch_size)
        outstanding += 1
        idx += 1

        sent_prog.update(n=present_batch_size)

        # Try to process at least one response per request
        outstanding = process_outstanding(outstanding >= max_outstanding, outstanding)

    tqdm.tqdm.write("All Requests Sent! Waiting for responses. Outstanding: {}.\n".format(outstanding))

    # Now process all outstanding requests
    while (outstanding > 0):
        outstanding = process_outstanding(True, outstanding)

    all_results_end = time.time()
    all_results_total = (all_results_end - all_results_start) * 1000.0

    print("-----------------------------")
    print("Total Time: {} ms".format(all_results_total))
    print("-----------------------------")

    print("-----------------------------")
    print("Total Inference Time = %0.2f for"
          "Sentences processed = %d" % (sum(time_list), len(eval_features)))
    print("Throughput Average (sentences/sec) = %0.2f" % (len(eval_features) / all_results_total * 1000.0))
    print("-----------------------------")

    if FLAGS.output_dir and FLAGS.predict_file:
        # When inferencing on a dataset, get inference statistics and write results to json file
        time_list.sort()

        avg = np.mean(time_list)
        cf_95 = max(time_list[:int(len(time_list) * 0.95)])
        cf_99 = max(time_list[:int(len(time_list) * 0.99)])
        cf_100 = max(time_list[:int(len(time_list) * 1)])
        print("-----------------------------")
        print("Summary Statistics")
        print("Batch size =", FLAGS.predict_batch_size)
        print("Sequence Length =", FLAGS.max_seq_length)
        print("Latency Confidence Level 95 (ms) =", cf_95 * 1000)
        print("Latency Confidence Level 99 (ms)  =", cf_99 * 1000)
        print("Latency Confidence Level 100 (ms)  =", cf_100 * 1000)
        print("Latency Average (ms)  =", avg * 1000)
        print("-----------------------------")


        output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
        output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

        write_predictions(eval_examples, eval_features, all_results,
                          FLAGS.n_best_size, FLAGS.max_answer_length,
                          FLAGS.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file,
                          FLAGS.version_2_with_negative, FLAGS.verbose_logging)
    else:
        # When inferencing on a single example, write best answer to stdout
        all_predictions, all_nbest_json, scores_diff_json = get_predictions(
                  eval_examples, eval_features, all_results,
                  FLAGS.n_best_size, FLAGS.max_answer_length,
                  FLAGS.do_lower_case, FLAGS.version_2_with_negative, 
                  FLAGS.verbose_logging)
        print("Context is: %s \n\nQuestion is: %s \n\nPredicted Answer is: %s" %(FLAGS.context, FLAGS.question, all_predictions[0]))


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  tf.compat.v1.app.run()

