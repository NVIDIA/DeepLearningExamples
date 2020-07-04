import json
import argparse
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import numpy as np

def smooth_moving_average(x, n):
    fil = np.ones(n)/n
    smoothed = np.convolve(x, fil, mode='valid')
    smoothed = np.concatenate((x[:n-1], smoothed), axis=0)
    
    return smoothed

def moving_stdev(x, n):
    fil = np.ones(n)/n
    avg_sqare = np.convolve(np.power(x, 2), fil, mode='valid')
    squared_avg = np.power(np.convolve(x, fil, mode='valid'), 2)
    var = avg_sqare - squared_avg
    stdev = np.sqrt(var)
    #pad first few values
    stdev = np.concatenate(([0]*(n-1), stdev), axis=0)
    
    return stdev

def get_plot(log):
    steps = [x[0] for x in log if isinstance(x[0], int)]
    values = [x[2] for x in log if isinstance(x[0], int)]
    return steps, values

def highlight_max_point(plot, color):
    point = max(zip(*plot), key=lambda x: x[1])
    plt.plot(point[0], point[1], 'bo-', color=color)
    plt.annotate("{:.2f}".format(point[1]), point)
    return point

def main(args):
    jlog = defaultdict(list)
    jlog['parameters'] = {}

    with open(args.log_file, 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line[5:])
            if line_dict['type'] == 'LOG':
                if line_dict['step'] == 'PARAMETER':
                    jlog['parameters'].update(line_dict['data'])
                elif line_dict['step'] == [] and 'training_summary' not in jlog:
                    jlog['training_summary']=line_dict['data']
                else:
                    for k, v in line_dict['data'].items():
                        jlog[k].append((line_dict['step'], line_dict['elapsedtime'], v))

    fig, ax1 = plt.subplots(figsize=(20,5))
    fig.suptitle(args.title, fontsize=16)
    ax1.set_xlabel('steps')
    ax1.set_ylabel('loss')

    # Define colors for specific curves
    VAL_LOSS_COLOR = 'blue'
    VAL_BLEU_COLOR = 'red'
    TEST_BLEU_COLOR = 'pink'

    # Plot smoothed loss curve
    steps, loss = get_plot(jlog['loss'])
    smoothed_loss = smooth_moving_average(loss, 150)
    stdev = moving_stdev(loss, 150)

    ax1.plot(steps, smoothed_loss, label='Training loss')
    ax1.plot(steps, smoothed_loss + stdev, '--', color='orange', linewidth=0.3, label='Stdev')
    ax1.plot(steps, smoothed_loss - stdev, '--', color='orange', linewidth=0.3)

    # Plot validation loss curve
    val_steps, val_loss = get_plot(jlog['val_loss'])
    ax1.plot(val_steps, val_loss, color='blue', label='Validation loss')

    min_val_loss_step = val_steps[np.argmin(val_loss)]
    ax1.axvline(min_val_loss_step, linestyle='dashed', color=VAL_LOSS_COLOR, linewidth=0.5, label='Validation loss minimum')

    # Plot BLEU curves
    ax2 = ax1.twinx()
    ax2.set_ylabel('BLEU')
    val_steps, val_bleu = get_plot(jlog['val_bleu'])
    ax2.plot(val_steps, val_bleu, color=VAL_BLEU_COLOR, label='Validation BLEU')
    mvb_step, _ =highlight_max_point((val_steps,val_bleu), color=VAL_BLEU_COLOR)

    # values to be labeled on plot
    max_val_bleu_step = val_steps[np.argmax(val_bleu)]
    max_val_bleu = val_bleu[val_steps.index(max_val_bleu_step)]
    min_loss_bleu = val_bleu[val_steps.index(min_val_loss_step)]


    if 'test_bleu' in jlog:
        test_steps, test_bleu = get_plot(jlog['test_bleu'])
        ax2.plot(val_steps, test_bleu, color=TEST_BLEU_COLOR, label='Test BLEU')
        highlight_max_point((test_steps, test_bleu), color=TEST_BLEU_COLOR)
    ax2.tick_params(axis='y')

    # Annotate points with highest BLEU score as well as those for minimal validation loss
    ax2.plot(min_val_loss_step, min_loss_bleu, 'bo-', color=VAL_BLEU_COLOR)
    ax2.annotate("{:.2f}".format(min_loss_bleu), (min_val_loss_step, min_loss_bleu))

    if 'test_bleu' in jlog:
        min_loss_test_bleu = test_bleu[val_steps.index(min_val_loss_step)] #BLEU score on test set when validation loss is minimal
        ax2.plot(min_val_loss_step, min_loss_test_bleu, 'bo-', color=TEST_BLEU_COLOR)
        ax2.annotate("{:.2f}".format(min_loss_test_bleu), (min_val_loss_step, min_loss_test_bleu))

        max_val_bleu_test = test_bleu[val_steps.index(max_val_bleu_step)] #BLEU score on test set when BLEU score on dev set is maximal
        ax2.plot(mvb_step, max_val_bleu_test, 'bo-', color=TEST_BLEU_COLOR)
        ax2.annotate("{:.2f}".format(max_val_bleu_test), (max_val_bleu_step, max_val_bleu_test))

    ax1.legend(loc='lower left', bbox_to_anchor=(1,0))
    ax2.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.grid()
    plt.savefig(args.output)

    # Produce json with training summary
    if args.dump_json:
        summary = OrderedDict()
        summary['args'] = OrderedDict(jlog['parameters'])
        summary['min_val_loss'] = min(val_loss)
        summary['max_val_bleu'] = max(val_bleu)
        summary['max_test_bleu'] = max(test_bleu)
        summary['final_values'] = jlog['training_summary']
        summary['avg_epoch_loss'] = [x.mean() for x in np.array_split(np.array(loss), jlog['parameters']['max_epoch'])]
        summary['min_val_loss_step'] = min_val_loss_step
        json.dump(summary, open(args.dump_json, 'w'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str)
    parser.add_argument('--log-file', type=str)
    parser.add_argument('--output' ,'-o', type=str)
    parser.add_argument('--dump-json', '-j', type=str)
    args = parser.parse_args()
    main(args)
