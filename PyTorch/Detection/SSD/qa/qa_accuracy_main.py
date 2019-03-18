# core imports
import os
import numpy as np

# pytorch imports
import torch
import torch.utils.data.distributed

# Apex imports
try:
    from apex.parallel.LARC import LARC
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")

# project imports
from src.train import train_loop
from main import train, make_parser
from src.logger import Logger
from qa.qa_utils import load_json, create_json_file, compare_acc, save_json

RESULT = None


def add_benchmark_args(parser):
    parser.add_argument('--benchmark-mode', type=str, default='epoch-accuracy',
                        choices=['full-accuracy', 'epoch-accuracy'], required=True)
    parser.add_argument('--benchmark-file', type=str, default=None, metavar='FILE',
                        help='path to the file with baselines', required=True)
    return parser


def main(args):
    if args.local_rank == 0:
        os.makedirs('./models', exist_ok=True)

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

    torch.backends.cudnn.benchmark = True

    if args.benchmark_mode == 'epoch-accuracy':
        args.epochs = 1

    train_loop_func = train_loop
    logger = Logger('Accuracy test', print_freq=10)

    args.evaluation = list(range(90))
    train(train_loop_func, logger, args)

    exit_code = 0
    if args.local_rank == 0:
        train_loss_results, val_acc_results, train_time_results = logger.print_results()
        print(train_time_results)
        print(train_loss_results)
        print(val_acc_results)
        measured_results = create_json_file(val_acc_results, train_loss_results, ngpus=8, bs=args.batch_size)
        save_json('/results/results.json', measured_results)
        print(measured_results)
        benchmark_results = load_json(args.benchmark_file)
        exit_code = compare_acc(measured_results, benchmark_results, args)
        exit(exit_code)


if __name__ == "__main__":
    parser = make_parser()
    parser = add_benchmark_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
