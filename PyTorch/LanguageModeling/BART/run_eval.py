# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import shutil
import datetime
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List
from json import JSONDecodeError

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import dllogger

from bart.configuration.configuration_bart import BartConfig
from bart.tokenization.tokenization_bart import BartTokenizer
from bart.modeling.modeling_bart import BartForConditionalGeneration, shift_tokens_right
from utils.utils import (
    calculate_bleu,
    calculate_rouge,
    Seq2SeqDataset,
    parse_numeric_n_bool_cl_kwargs,
    use_task_specific_params,
    encode_line,
    load_json,
    lmap,
    chunks,
    write_txt_file,
    save_json,
    format_step)
import utils.distributed_utils


logger = getLogger(__name__)


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def distill(layers, num_layers):
    sft_layers = nn.ModuleList()
    for i in range(num_layers):
        sft_layers.append(layers[i])

    # delete unnecessary layers
    delete_layers = [i for i in range(num_layers, len(layers))]
    for i in range(len(delete_layers)):
        del layers[delete_layers[i] - i]

    return sft_layers

def distill_sft(model, num_layers, do_encoder=False, do_decoder=False):
    if do_encoder:
        layers = model.model.encoder.layers
        sft_layers = distill(layers, num_layers)
        model.model.encoder.layers = sft_layers

    if do_decoder:
        layers = model.model.decoder.layers
        sft_layers = distill(layers, num_layers)
        model.model.decoder.layers = sft_layers

    return model


def generate_summaries_or_translations(
    data_dir: str,
    out_dir: str,
    model_path: str,
    config_path: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    bf16=False,
    pre_ln=False,
    task="summarization",
    prefix=None,
    max_source_length=1024,
    max_target_length=142,
    eval_beams=5,
    eval_max_gen_length=142,
    n_obs=-1,
    type_path="test",
    num_return_sequences=1,
    distill=None,
    num_layers=None,
    do_encoder=False,
    do_decoder=False,
    **generate_kwargs,
) -> Dict:

    out_dir = Path(out_dir)
    save_path = out_dir.joinpath(f"rank_{utils.distributed_utils.get_rank()}_output.json")

    if num_return_sequences > eval_beams:
        eval_beams = num_return_sequences

    ### Define BART model
    # Config from "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-cnn/config.json
    # Vocab modified to 50265 to be consistent with facebook/bart-large default
    config = BartConfig(**json.load(open(config_path, "r")))
    if fp16:
        config.dtype = torch.float16
    elif bf16:
        config.dtype = torch.bfloat16
    else:
        config.dtype = None
    config.pre_ln = pre_ln

    model = BartForConditionalGeneration.from_pretrained(model_path, config=config).to(device)

    # if distilling, change model
    if distill == "sft":
        model = distill_sft(model, num_layers, do_encoder, do_decoder)

    if fp16:
        model = model.half()
    elif bf16:
        model = model.bfloat16()
    model.eval()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.

    start_time = time.time()
    # update config with task specific params
    use_task_specific_params(model, task)
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""

    ds = Seq2SeqDataset(tokenizer, data_dir, max_source_length, max_target_length, type_path=type_path,
        n_obs=n_obs, prefix=prefix)

    # I set shuffle=True for a more accurate progress bar.
    # If all the longest samples are first, the prog bar estimate is too high at the beginning.
    is_distributed = True if utils.distributed_utils.get_world_size() > 1 else False
    sampler = ds.make_sortish_sampler(batch_size, distributed=is_distributed, add_extra_examples=False, shuffle=True)
    data_loader = DataLoader(ds, sampler=sampler, batch_size=batch_size, collate_fn=ds.collate_fn)

    results = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            torch.cuda.synchronize()
            t0 = time.time()

            summaries = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                use_cache=True,
                num_return_sequences=num_return_sequences,
                num_beams=eval_beams,
                max_length=eval_max_gen_length,
                num_beam_groups=1, output_scores=False,
                return_dict_in_generate=False,
                encoder_no_repeat_ngram_size=0,
                diversity_penalty=0.0,
                **generate_kwargs,
            )
            preds = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ids = batch["ids"]
            if num_return_sequences > 1:
                preds = chunks(preds, num_return_sequences)  # batch size chunks, each of size num_return_seq

            torch.cuda.synchronize()
            eval_time = time.time() - t0
            for i, pred in enumerate(preds):
                store_time = eval_time if i == 0 else None #only store latency for element 0 of every batch
                results.append(dict(pred=pred, id=ids[i].item(), eval_time=store_time))

    save_json(results, save_path)
    runtime = int(time.time() - start_time)  # seconds
    num_replicas = sampler.num_replicas if is_distributed else 1
    n_obs = len(results)
    return results, num_replicas, dict(n_obs=n_obs, eval_only_runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4))


def datetime_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_generate(verbose=True):
    """

    Takes input text, generates output, and then using reference calculates the BLEU scores.

    The results are saved to a file and returned to the caller, and printed out unless ``verbose=False`` is passed.

    Args:
        verbose (:obj:`bool`, `optional`, defaults to :obj:`True`): print results to stdout

    Returns:
        a tuple: ``(scores, params}``
        - ``scores``: a dict of scores data ``{'bleu': 39.6501, 'n_obs': 2000, 'runtime': 186, 'seconds_per_sample': 0.093}``
        - ``params``: a dict of custom params, e.g. ``{'num_beams': 5, 'length_penalty': 0.8}``
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="like facebook/bart-large-cnn or path to ckpt")
    parser.add_argument("config_path", type=str, help="path to config")
    parser.add_argument("data_dir", type=str, help="like cnn_dm/test.source")
    parser.add_argument("save_path", type=str, help="where to save summaries")
    parser.add_argument("--type_path", type=str, required=False, default="test", help="like cnn_dm/test.target")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        "--prefix", type=str, required=False, default=None, help="will be added to the begininng of src examples"
    )
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument(
        "--n_obs", type=int, default=None, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument(
        "--num_return_sequences", type=int, default=1, required=False, help="How many sequences to return"
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--dump-args", action="store_true", help="print the custom hparams with the results")
    parser.add_argument(
        "--info",
        nargs="?",
        type=str,
        const=datetime_now(),
        help="use in conjunction w/ --dump-args to print with the results whatever other info you'd like, e.g. lang=en-ru. If no value is passed, the current datetime string will be used.",
    )
    parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
    parser.add_argument("--eval_beams", type=int, default=None, required=False, help="# beams to use. 0 corresponds to not using beam search.")
    parser.add_argument(
        "--max_source_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=142,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--sync_timeout",
        type=int,
        default=600,
        required=False,
        help="How long should master process wait for other processes to finish.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--json-summary', type=str, default="results/dllogger.json",
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument('--distill', type=str, default=None, help="string indicating how model is distilled, only sft supported", choices=["sft",None])
    parser.add_argument('--layers', type=str, default=None, help="string indicating which teacher layers remain, split by '-' (ex. 0-6-11)")
    parser.add_argument('--do_encoder', action="store_true", default=False, help="if true encoder distilled")
    parser.add_argument('--do_decoder', action="store_true", default=False, help="if true decoder distilled")
    parser.add_argument("--pre_ln",
        default=False,
        action='store_true',
        help="Whether to use Pre-LN architecture."
    )

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank',  type=int,
                      default=os.getenv('LOCAL_RANK', 0),
                      help='Used for multi-process training.')

    start_time = time.time()

    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    parsed_args = parse_numeric_n_bool_cl_kwargs(rest)

    if args.local_rank <= 0:
        print(args)
        print(rest)

    # Initialize device and distributed backend
    utils.distributed_utils.init_distributed(args.device == "cuda")
    if utils.distributed_utils.get_world_size() > 1:
        utils.distributed_utils.set_affinity(args.local_rank)
        torch.cuda.set_device(args.local_rank)

    if Path(args.json_summary).exists():
        warnings.warn(f"json_summary {args.json_summary} will be overwritten unless you type ctrl-c.")

    if utils.distributed_utils.get_rank() == 0:
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])

    if parsed_args and verbose:
        print(f"parsed the following generate kwargs: {parsed_args}")

    Path(args.save_path).parent.mkdir(exist_ok=True)
    json_save_path = Path(args.save_path + "/tmp")
    Path(json_save_path).mkdir(exist_ok=True)  # this handles locking.

    if args.layers:
        num_layers = len(args.layers.split('-'))
    else:
        num_layers = None

    results, num_replicas, runtime_metrics = generate_summaries_or_translations(
        args.data_dir,
        json_save_path,
        args.model_path,
        args.config_path,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        bf16=args.bf16,
        pre_ln=args.pre_ln,
        task=args.task,
        prefix=args.prefix,
        eval_beams=args.eval_beams,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        eval_max_gen_length=args.eval_max_gen_length,
        n_obs=args.n_obs,
        type_path=args.type_path,
        num_return_sequences=args.num_return_sequences,
        distill=args.distill,
        num_layers=num_layers,
        do_encoder=args.do_encoder,
        do_decoder=args.do_decoder,
        **parsed_args,
    )


    if args.local_rank <= 0:
        save_path = Path(args.save_path)
        save_path.mkdir(exist_ok=True)
        partial_results = gather_results_from_each_node(num_replicas, json_save_path, args.sync_timeout)
        preds, time_list = combine_partial_results(partial_results)
        if args.num_return_sequences > 1:
            save_path = save_path.joinpath("pseudolabel_results.json")
            print(f"Saving aggregated results at {save_path}, intermediate in {json_save_path}/")
            save_json(preds, save_path)
            return
        tgt_file = Path(args.data_dir).joinpath(args.type_path + ".target")
        labels = [x.rstrip() for x in open(tgt_file).readlines()][: len(preds)]

        # Calculate metrics, save metrics,  and save _generations.txt
        calc_bleu = "translation" in args.task
        score_fn = calculate_bleu if calc_bleu else calculate_rouge
        metric_name = "bleu" if calc_bleu else "rouge"
        metrics: Dict = score_fn(preds, labels)
        metrics["n_obs"] = len(preds)
        runtime = time.time() - start_time
        metrics["seconds_per_sample"] = round(runtime / metrics["n_obs"], 4)
        metrics["n_gpus"] = num_replicas
        metrics.update(runtime_metrics)

        time_list.sort()
        metrics["inference_latency_mean"] = np.mean(time_list)
        metrics["inference_latency_conf_50"] = max(time_list[:int(len(time_list) * 0.50)])
        metrics["inference_latency_conf_90"] = max(time_list[:int(len(time_list) * 0.90)])
        metrics["inference_latency_conf_95"] = max(time_list[:int(len(time_list) * 0.95)])
        metrics["inference_latency_conf_99"] = max(time_list[:int(len(time_list) * 0.99)])
        metrics["inference_latency_conf_100"] = max(time_list[:int(len(time_list) * 1)])
        metrics["inference_throughput_mean"] = len(preds) * 1.0 / sum(time_list)


        metrics_save_path = save_path.joinpath(f"{args.type_path}_{metric_name}.json")
        save_json(metrics, metrics_save_path, indent=None)
        dllogger.log(step=tuple(), data=metrics)
        print(metrics)
        write_txt_file(preds, save_path.joinpath(f"{args.type_path}_generations.txt"))
        if args.debug:
            write_txt_file(labels, save_path.joinpath(f"{args.type_path}.target"))
        else:
            shutil.rmtree(json_save_path)

    dllogger.flush()


def combine_partial_results(partial_results) -> List:
    """Concatenate partial results into one file, then sort it by id."""
    records = []
    for partial_result in partial_results:
        records.extend(partial_result)
    records = list(sorted(records, key=lambda x: x["id"]))
    preds = [x["pred"] for x in records]
    eval_time = [x["eval_time"] for x in records if x["eval_time"] is not None]
    return preds, eval_time


def gather_results_from_each_node(num_replicas, save_path, timeout) -> List[Dict[str, List]]:
    # WAIT FOR lots of .json files
    start_wait = time.time()
    logger.info("waiting for all nodes to finish")
    json_data = None
    while (time.time() - start_wait) < timeout:
        json_files = list(save_path.glob("rank_*.json"))
        if len(json_files) < num_replicas:
            continue
        try:
            # make sure all json files are fully saved
            json_data = lmap(load_json, json_files)
            return json_data
        except JSONDecodeError:
            continue
    else:
        raise TimeoutError("Rank 0 gave up on waiting for other processes")
    # Unreachable


if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME $DATA_DIR/test.source $save_path/test_translations.txt --reference_path $DATA_DIR/test.target --task translation $@
    run_generate(verbose=True)
