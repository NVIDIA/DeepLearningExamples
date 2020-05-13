#!/usr/bin/env python3
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

import time
import json
import ctypes
import argparse
import collections
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import helpers.tokenization as tokenization
import helpers.data_processing as dp

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BERT QA Inference')
    parser.add_argument('-e', '--engine',
            help='Path to BERT TensorRT engine')
    parser.add_argument('-p', '--passage', nargs='*',
            help='Text for paragraph/passage for BERT QA',
            default='')
    parser.add_argument('-pf', '--passage-file',
            help='File containing input passage',
            default='')
    parser.add_argument('-q', '--question', nargs='*',
            help='Text for query/question for BERT QA',
            default='')
    parser.add_argument('-qf', '--question-file',
            help='File containiner input question',
            default='')
    parser.add_argument('-sq', '--squad-json',
            help='SQuAD json file',
            default='')
    parser.add_argument('-o', '--output-prediction-file',
            help='Output prediction file for SQuAD evaluation',
            default='./predictions.json')
    parser.add_argument('-v', '--vocab-file',
            help='Path to file containing entire understandable vocab')
    parser.add_argument('-s', '--sequence-length',
            help='The sequence length to use. Defaults to 128',
            default=128, type=int)
    parser.add_argument('--max-query-length',
            help='The maximum length of a query in number of tokens. Queries longer than this will be truncated',
            default=64, type=int)
    parser.add_argument('--max-answer-length',
            help='The maximum length of an answer that can be generated',
            default=30, type=int)
    parser.add_argument('--n-best-size',
            help='Total number of n-best predictions to generate in the nbest_predictions.json output file',
            default=20, type=int)
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    paragraph_text = None
    squad_examples = None
    output_prediction_file = None

    if not args.passage == '':
        paragraph_text = ' '.join(args.passage)
    elif not args.passage_file == '':
        f = open(args.passage_file, 'r')
        paragraph_text = f.read()
    elif not args.squad_json == '':
        squad_examples = dp.read_squad_json(args.squad_json)
        output_prediction_file = args.output_prediction_file
    else:
        paragraph_text = input("Paragraph: ")

    question_text = None
    if not args.question == '':
        question_text = ' '.join(args.question)
    elif not args.question_file == '':
        f = open(args.question_file, 'r')
        question_text = f.read()

    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    # When splitting up a long document into chunks, how much stride to take between chunks.
    doc_stride = 128
    # The maximum total input sequence length after WordPiece tokenization.
    # Sequences longer than this will be truncated, and sequences shorter
    max_seq_length = args.sequence_length

    def question_features(tokens, question):
        # Extract features from the paragraph and question
        return dp.convert_example_to_features(tokens, question, tokenizer, max_seq_length, doc_stride, args.max_query_length)

    # Import necessary plugins for BERT TensorRT
    ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)

    # The first context created will use the 0th profile. A new context must be created
    # for each additional profile needed. Here, we only use batch size 1, thus we only need the first profile.
    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, \
        runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

        # We always use batch size 1.
        input_shape = (max_seq_length, 1)
        input_nbytes = trt.volume(input_shape) * trt.int32.itemsize

        # Specify input shapes. These must be within the min/max bounds of the active profile (0th profile in this case)
        # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
        for binding in range(3):
            context.set_binding_shape(binding, input_shape)
        assert context.all_binding_shapes_specified

        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        # Allocate device memory for inputs.
        d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)

        def inference(features, tokens):
            global h_output

            _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
                    "NetworkOutput",
                    ["start_logits", "end_logits", "feature_index"])
            networkOutputs = []

            eval_time_elapsed = 0
            for feature_index, feature in enumerate(features):
                # Copy inputs
                input_ids = cuda.register_host_memory(np.ascontiguousarray(feature.input_ids.ravel()))
                segment_ids = cuda.register_host_memory(np.ascontiguousarray(feature.segment_ids.ravel()))
                input_mask = cuda.register_host_memory(np.ascontiguousarray(feature.input_mask.ravel()))

                eval_start_time = time.time()
                cuda.memcpy_htod_async(d_inputs[0], input_ids, stream)
                cuda.memcpy_htod_async(d_inputs[1], segment_ids, stream)
                cuda.memcpy_htod_async(d_inputs[2], input_mask, stream)

                # Run inference
                context.execute_async_v2(bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
                # Synchronize the stream
                stream.synchronize()
                eval_time_elapsed += (time.time() - eval_start_time)

                # Transfer predictions back from GPU
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()

                for index, batch in enumerate(h_output):
                    # Data Post-processing
                    networkOutputs.append(_NetworkOutput(
                        start_logits = np.array(batch.squeeze()[:, 0]),
                        end_logits = np.array(batch.squeeze()[:, 1]),
                        feature_index = feature_index
                        ))

            eval_time_elapsed /= len(features)

            prediction, nbest_json, scores_diff_json = dp.get_predictions(tokens, features,
                    networkOutputs, args.n_best_size, args.max_answer_length)

            return eval_time_elapsed, prediction, nbest_json

        def print_single_query(eval_time_elapsed, prediction, nbest_json):
            print("------------------------")
            print("Running inference in {:.3f} Sentences/Sec".format(1.0/eval_time_elapsed))
            print("------------------------")

            print("Answer: '{}'".format(prediction))
            print("With probability: {:.3f}".format(nbest_json[0]['probability'] * 100.0))

        if squad_examples:
            all_predictions = collections.OrderedDict()

            for example in squad_examples:
                features = question_features(example.doc_tokens, example.question_text)
                eval_time_elapsed, prediction, nbest_json = inference(features, example.doc_tokens)
                all_predictions[example.id] = prediction

            with open(output_prediction_file, "w") as f:
                f.write(json.dumps(all_predictions, indent=4))
                print("\nOutput dump to {}".format(output_prediction_file))
        else:
            # Extract tokecs from the paragraph
            doc_tokens = dp.convert_doc_tokens(paragraph_text)

            if question_text:
                print("\nPassage: {}".format(paragraph_text))
                print("\nQuestion: {}".format(question_text))

                features = question_features(doc_tokens, question_text)
                eval_time_elapsed, prediction, nbest_json = inference(features, doc_tokens)
                print_single_query(eval_time_elapsed, prediction, nbest_json)

            else:
                # If no question text is provided, loop until the question is 'exit'
                EXIT_CMDS = ["exit", "quit"]
                question_text = input("Question (to exit, type one of {:}): ".format(EXIT_CMDS))

                while question_text.strip() not in EXIT_CMDS:
                    features = question_features(doc_tokens, question_text)
                    eval_time_elapsed, prediction, nbest_json = inference(features, doc_tokens)
                    print_single_query(eval_time_elapsed, prediction, nbest_json)
                    question_text = input("Question (to exit, type one of {:}): ".format(EXIT_CMDS))
