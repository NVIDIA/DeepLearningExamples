# core imports
import os
import numpy as np
import json
from pprint import pprint
import time

# pytorch imports
import torch
import torch.utils.data.distributed
from torch.autograd import Variable


# Apex imports
try:
    from apex.parallel.LARC import LARC
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")

# project imports
from main import train, make_parser
from src.logger import BenchLogger
# from src.train import benchmark_inference_loop, benchmark_train_loop

from SSD import _C as C

RESULT = None


def add_benchmark_args(parser):
    parser.add_argument('--benchmark-mode', type=str, choices=['training', 'inference'],
                        default='inference', required=True)
    parser.add_argument('--results-file', default='experiment_raport.json', type=str,
                        help='file in which to store JSON experiment raport')
    parser.add_argument('--benchmark-file', type=str, default=None, metavar='FILE',
                        help='path to the file with baselines')
    return parser

def benchmark_train_loop(model, loss_func, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    start_time = None
    # tensor for results
    result = torch.zeros((1,)).cuda()
    for i, data in enumerate(loop(train_dataloader)):
        if i >= args.benchmark_warmup:
            start_time = time.time()

        img = data[0][0][0]
        bbox = data[0][1][0]
        label = data[0][2][0]
        label = label.type(torch.cuda.LongTensor)
        bbox_offsets = data[0][3][0]
        # handle random flipping outside of DALI for now
        bbox_offsets = bbox_offsets.cuda()
        img, bbox = C.random_horiz_flip(img, bbox, bbox_offsets, 0.5, False)

        if not args.no_cuda:
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()
            bbox_offsets = bbox_offsets.cuda()
        img.sub_(mean).div_(std)

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")
            continue
        bbox, label = C.box_encoder(N, bbox, bbox_offsets, label, encoder.dboxes.cuda(), 0.5)

        M = bbox.shape[0] // N
        bbox = bbox.view(N, M, 4)
        label = label.view(N, M)





        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()

        trans_bbox = bbox.transpose(1, 2).contiguous().cuda()

        if not args.no_cuda:
            label = label.cuda()
        gloc = Variable(trans_bbox, requires_grad=False)
        glabel = Variable(label, requires_grad=False)

        loss = loss_func(ploc, plabel, gloc, glabel)



        # loss scaling
        if args.fp16:
            if args.amp:
                with optim.scale_loss(loss) as scale_loss:
                    scale_loss.backward()
            else:
                optim.backward(loss)
        else:
            loss.backward()

        optim.step()
        optim.zero_grad()
        iteration += 1

        # reduce all results from every gpu
        if i >= args.benchmark_warmup + args.benchmark_iterations:
            result.data[0] = logger.print_result()
            if args.N_gpu > 1:
                torch.distributed.reduce(result, 0)
            if args.local_rank == 0:
                global RESULT
                RESULT = float(result.data[0])
            return

        if i >= args.benchmark_warmup:
            logger.update(args.batch_size, time.time() - start_time)

def loop(dataloader):
    while True:
        for data in dataloader:
            yield data

def benchmark_inference_loop(model, loss_func, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    assert args.N_gpu == 1, 'Inference benchmark only on 1 gpu'
    start_time = None
    model.eval()
    i=-1
    dataloader = loop(val_dataloader)
    while True:
        i+=1
        with torch.no_grad():
            torch.cuda.synchronize()
            if i >= args.benchmark_warmup:
                start_time = time.time()
            data = next(dataloader)

            img = data[0]

            if not args.no_cuda:
                img = img.cuda()

            if args.fp16:
                img = img.half()

            img.sub_(mean).div_(std)
            img = Variable(img, requires_grad=False)
            _ = model(img)
            torch.cuda.synchronize()

            if i >= args.benchmark_warmup + args.benchmark_iterations:
                global RESULT
                RESULT = logger.print_result()
                return

            if i >= args.benchmark_warmup:
                logger.update(args.batch_size, time.time() - start_time)


def main(args):
    if args.local_rank == 0:
        os.makedirs('./models', exist_ok=True)

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

    torch.backends.cudnn.benchmark = True

    if args.benchmark_mode == 'training':
        train_loop_func = benchmark_train_loop
        logger = BenchLogger('Training benchmark')
    else:
        train_loop_func = benchmark_inference_loop
        logger = BenchLogger('Inference benchmark')

    args.epochs = 1

    train(train_loop_func, logger, args)

    if args.local_rank == 0:
        global RESULT
        with open(args.results_file) as f:
            results = json.load(f)
        results['metrics'][str(args.N_gpu)][str(args.batch_size)] = {'images_per_second': RESULT}
        pprint(results)

        with open(args.results_file, 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":
    parser = make_parser()
    parser = add_benchmark_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
