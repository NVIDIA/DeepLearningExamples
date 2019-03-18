import argparse
import subprocess

from qa.qa_utils import compare_benchmarks, load_json, save_json, OKBLUE, ENDC, FAIL



# parsing
def parse_testscript_args():
    parser = argparse.ArgumentParser(description='PyTorch Benchmark Tests')
    parser.add_argument('--bs', default=[1], type=int, nargs='+')
    parser.add_argument('--ngpus', default=[1], type=int, nargs='+')
    parser.add_argument('--benchmark-mode', default='training', choices=['training', 'inference'],
                        help='benchmark training or inference', required=True)
    parser.add_argument('--bench-iterations', type=int, default=20, metavar='N',
                        help='Run N iterations while benchmarking (ignored when training and validation)')
    parser.add_argument('--bench-warmup', type=int, default=10, metavar='N',
                        help='Number of warmup iterations for benchmarking')
    parser.add_argument('--fp16', action='store_true', help='Run model in mixed precision.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--data', type=str, metavar='<PATH>', required=True,
                        help='path to the dataset')
    parser.add_argument('--results-file', default='experiment_raport.json', type=str,
                        help='file in which to store JSON experiment raport')
    parser.add_argument('--benchmark-file', type=str, metavar='FILE', required=True,
                        help='path to the file with baselines')
    return parser.parse_args()


# job command
command_template = 'python3 {launcher} qa/qa_perf_main.py --bs {bs} --ebs {bs} ' \
                   '--benchmark-mode {mode} --benchmark-warmup {bw} --benchmark-iterations {bi} {fp16} ' \
                   '--backbone resnet50 --seed 1 --data {data} --results-file {results_file} --benchmark-file {benchmark_file}'

if __name__ == '__main__':
    args = parse_testscript_args()

    fp16 = '--fp16' if args.fp16 else ''

    # create results json file
    # todo: maybe some template json file?
    results = {'ngpus': args.ngpus,
               'bs': args.bs,
               'metric_keys': ['images_per_second'],
               'metrics': {}}

    for gpu in args.ngpus:
        results['metrics'][str(gpu)] = {}
        for bs in args.bs:
            results['metrics'][str(gpu)][str(bs)] = {'images_per_second': None}

    save_json(args.results_file, results)

    # run qa_perf_main.py tests one by one
    for gpu in args.ngpus:
        launcher = '' if gpu == 1 else '-m torch.distributed.launch --nproc_per_node={}'.format(gpu)
        for bs in args.bs:
            print('#' * 80)
            command = command_template.format(launcher=launcher, bs=bs, workers=args.workers, mode=args.benchmark_mode,
                                              bw=args.bench_warmup, bi=args.bench_iterations, fp16=fp16,
                                              data=args.data, results_file=args.results_file,
                                              benchmark_file=args.benchmark_file)

            print('Running "{}"'.format(command))

            process = subprocess.Popen(command, shell=True)
            output, error = process.communicate()

            if error is not None:
                print(FAIL + 'Program exited with status {}. Data has not been collected'.format(error) + ENDC)
            # elif results['metrics'][str(gpu)][str(bs)]['images_per_second'] is None:
            #     print(WARNING + 'Program did not end sucessfully. Data has not been collected.' + ENDC)
            else:
                print(OKBLUE + 'Program ended sucessfully. Data has been collected.' + ENDC)

    results_data = load_json(args.results_file)
    benchmark_data = load_json(args.benchmark_file)
    exit_code = compare_benchmarks(results_data, benchmark_data, args, 0.16 if args.benchmark_mode == 'inference' else 0.1)
    print(exit_code)
    exit(exit_code)
