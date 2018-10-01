import argparse
import json
from pathlib import Path
from pprint import pprint


def parse_arguments():
    parser = argparse.ArgumentParser(description='summary extractor')
    parser.add_argument('filename', type=Path,
                        help='path to logfile')
    parser.add_argument('-H', '--human-readable', action='store_const', const=True, default=False,
                        help='human readable')
    parser.add_argument('--csv', action='store_const', const=True, default=False,
                        help='print in csv format')
    return parser.parse_args()

def extract_summary(content):
    train_summary = []
    eval_summary = []

    current_epoch = -1
    for line in content.splitlines():
        words = line.split()
        if line.startswith('Train summary:'):
            epoch = int(words[3].strip('[]'))
            loss = float(words[11])
            top1 = float(words[13])
            top5 = float(words[15])

            current_epoch += 1
            assert epoch == current_epoch

            train_summary.append({'loss': loss, 'top1': top1, 'top5': top5})

        if line.startswith('Eval summary:'):
            loss = float(words[7])
            top1 = float(words[9])
            top5 = float(words[11])

            eval_summary.append({'loss': loss, 'top1': top1, 'top5': top5})

    return train_summary, eval_summary

def main(args):
    with open(str(args.filename)) as file:
        content = file.read()

    train_summary, eval_summary = extract_summary(content)


    if args.human_readable:
        print('Train summary:')
        pprint(train_summary)
        print('Eval summary:')
        pprint(eval_summary)
    elif args.csv:
        print('train_loss', 'train_top1', 'train_top5',
              'eval_loss', 'eval_top1', 'eval_top5',
              sep=',', end=',\n')
        for summaries in zip(train_summary, eval_summary):
            for summary in summaries:
                print(summary['loss'], summary['top1'], summary['top5'], sep=',', end=',')
            print()
    else:
        result = {'train': train_summary, 'eval': eval_summary}
        print(json.dumps(result))

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
