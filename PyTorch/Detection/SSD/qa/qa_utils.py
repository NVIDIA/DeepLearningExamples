import json

# terminal stdout colors
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'


# load results and benchmark
def load_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data


def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f)


# compare func
def compare(measured_value, true_value, pmargin=0.1):
    assert 0 < pmargin < 1, 'Margin should be in range [0, 1]'
    return (1 - pmargin) * true_value < measured_value


# compare 2 benchmark json files
def compare_benchmarks(results, benchmark, args, pmargin=0.1):
    # sanity check
    for metric in results['metric_keys']:
        if metric not in benchmark['metric_keys']:
            assert False, "You want to compare {} metric which doesn't appear in benchmark file".format(metric)

    assert len(args.bs) <= len(benchmark['bs']), 'len(args.bs) <= len(benchmark["bs"] ({} <= {})'.format(len(args.bs),
                                                                                                         len(benchmark[
                                                                                                                 'bs']))
    assert len(args.bs) == len(results['bs']), 'len(args.bs) <= len(results["bs"] ({} == {})'.format(len(args.bs),
                                                                                                     len(results['bs']))
    for bs in results['bs']:
        if bs not in benchmark['bs']:
            assert False, "You want to compare batch size = {} which doesn't appear in benchmark file".format(bs)

    assert len(args.ngpus) <= len(benchmark['ngpus']), 'len(args.ngpus) <= len(benchmark["ngpus"]) ({} <= {})'.format(
        len(args.bs), len(benchmark['ngpus']))
    assert len(args.ngpus) == len(results['ngpus']), 'len(args.ngpus) == len(results["ngpus"]) ({} == {})'.format(
        len(args.bs), len(results['ngpus']))
    for gpu in results['ngpus']:
        if gpu not in benchmark['ngpus']:
            assert False, "You want to compare {} gpus results which don't appear in benchmark file".format(gpu)

    # compare measured numbers with benchmark
    exit = 0
    for metric in results['metric_keys']:
        for gpu in results['ngpus']:
            for bs in results['bs']:
                measured_metric = results['metrics'][str(gpu)][str(bs)][metric]
                ground_truth_metric = benchmark['metrics'][str(gpu)][str(bs)][metric]
                ok = compare(measured_metric, ground_truth_metric, pmargin)
                if ok:
                    print(OKGREEN + 'BENCHMARK PASSED: metric={} gpu={} bs={}'.format(metric, gpu, bs) + ENDC)
                else:
                    print(FAIL + 'BENCHMARK NOT PASSED: metric={} gpu={} bs={}'.format(metric, gpu, bs) + ENDC)
                    exit = 1
    return exit

# compare 2 benchmark json files
def compare_acc(results, benchmark, args):
    # sanity check
    for metric in results['metric_keys']:
        if metric not in benchmark['metric_keys']:
            assert False, "You want to compare {} metric which doesn't appear in benchmark file".format(metric)

    for bs in results['bs']:
        if bs not in benchmark['bs']:
            assert False, "You want to compare batch size = {} which doesn't appear in benchmark file".format(bs)

    for gpu in results['ngpus']:
        if gpu not in benchmark['ngpus']:
            assert False, "You want to compare {} gpus results which don't appear in benchmark file".format(gpu)

    # compare measured numbers with benchmark
    for i, (result, ground_truth) in enumerate(zip(results['metrics']['val.acc'], benchmark['metrics']['val.acc'])):
        if i > 43: # before first decay accuracy tends to vary more than 15% at ~30th epoch
            if ground_truth * 0.9 > result:
                print(FAIL + 'ACCURACY TEST NOT PASSED' + ENDC)
                return 1

    # compare measured numbers with benchmark
    for i, (result, ground_truth) in enumerate(zip(results['metrics']['train.loss'], benchmark['metrics']['train.loss'])):
        if i > 43:
            if ground_truth * 1.1 < result:
                print(FAIL + 'LOSS TEST NOT PASSED' + ENDC)
                return 1

    print(OKGREEN + 'ACCURACY TEST PASSED' + ENDC)
    return 0

def create_json_file(val_acc_results, train_loss_results, ngpus=8, bs=32):
    results = {"ngpus": [ngpus],
               "bs": [bs],
               "metric_keys": ["train.loss", "val.acc"],
               "metrics": {
                   "train.loss": [],
                   "val.acc": []
               }
               }

    for i, ((epoch1, acc), (epoch2, loss)) in enumerate(zip(val_acc_results, train_loss_results)):
        assert i == epoch1 == epoch2
        results['metrics']['train.loss'].append(loss)
        results['metrics']['val.acc'].append(acc)

    return results
