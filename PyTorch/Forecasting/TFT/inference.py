# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
import numpy as np
import pickle
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modeling import TemporalFusionTransformer
from configuration import ElectricityConfig
from data_utils import TFTDataset
from utils import PerformanceMeter
from criterions import qrisk
import dllogger
from log_helper import setup_logger
from torch.cuda import amp

def _unscale_per_id(config, values, ids, scalers):
    num_horizons = config.example_length - config.encoder_length + 1
    flat_values = pd.DataFrame(
            values,
            columns=[f't{j}' for j in range(num_horizons - values.shape[1], num_horizons)]
            )
    flat_values['id'] = ids
    df_list = []
    for idx, group in flat_values.groupby('id'):
        scaler = scalers[idx]
        group_copy = group.copy()
        for col in group_copy.columns:
            if not 'id' in col:
                _col = np.expand_dims(group_copy[col].values, -1)
                _t_col = scaler.inverse_transform(_col)[:,-1]
                group_copy[col] = _t_col
        df_list.append(group_copy)
    flat_values = pd.concat(df_list, axis=0)

    flat_values = flat_values[[col for col in flat_values if not 'id' in col]]
    return flat_values.values

def _unscale(config, values, scaler):
    num_horizons = config.example_length - config.encoder_length + 1
    flat_values = pd.DataFrame(
            values,
            columns=[f't{j}' for j in range(num_horizons - values.shape[1], num_horizons)]
            )
    for col in flat_values.columns:
        if not 'id' in col:
            _col = np.expand_dims(flat_values[col].values, -1)
            _t_col = scaler.inverse_transform(_col)[:,-1]
            flat_values[col] = _t_col

    flat_values = flat_values[[col for col in flat_values if not 'id' in col]]
    return flat_values.values

def predict(args, config, model, data_loader, scalers, cat_encodings, extend_targets=False):
    model.eval()
    predictions = []
    targets = []
    ids = []
    perf_meter = PerformanceMeter(benchmark_mode=not args.disable_benchmark)
    n_workers = args.distributed_world_size if hasattr(args, 'distributed_world_size') else 1
    
    with torch.jit.fuser("fuser2"):
        for step, batch in enumerate(data_loader):
            perf_meter.reset_current_lap()
            with torch.no_grad():
                batch = {key: tensor.cuda() if tensor.numel() else None for key, tensor in batch.items()}
                ids.append(batch['id'][:,0,:])
                targets.append(batch['target'])
                predictions.append(model(batch).float())

            perf_meter.update(args.batch_size * n_workers,
                exclude_from_total=step in [0, 1, 2, len(data_loader)-1])

    targets = torch.cat(targets, dim=0).cpu().numpy()
    if not extend_targets:
        targets = targets[:,config.encoder_length:,:] 
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    
    if config.scale_per_id:
        ids = torch.cat(ids, dim=0).cpu().numpy()

        unscaled_predictions = np.stack(
                [_unscale_per_id(config, predictions[:,:,i], ids, scalers) for i in range(len(config.quantiles))], 
                axis=-1)
        unscaled_targets = np.expand_dims(_unscale_per_id(config, targets[:,:,0], ids, scalers), axis=-1)
    else:
        ids = None
        unscaled_predictions = np.stack(
                [_unscale(config, predictions[:,:,i], scalers['']) for i in range(len(config.quantiles))], 
                axis=-1)
        unscaled_targets = np.expand_dims(_unscale(config, targets[:,:,0], scalers['']), axis=-1)

    return unscaled_predictions, unscaled_targets, ids, perf_meter

def visualize_v2(args, config, model, data_loader, scalers, cat_encodings):
    unscaled_predictions, unscaled_targets, ids, _ = predict(args, config, model, data_loader, scalers, cat_encodings, extend_targets=True)

    num_horizons = config.example_length - config.encoder_length + 1
    pad = unscaled_predictions.new_full((unscaled_targets.shape[0], unscaled_targets.shape[1] - unscaled_predictions.shape[1], unscaled_predictions.shape[2]), fill_value=float('nan'))
    pad[:,-1,:] = unscaled_targets[:,-num_horizons,:]
    unscaled_predictions = torch.cat((pad, unscaled_predictions), dim=1)

    ids = torch.from_numpy(ids.squeeze())
    joint_graphs = torch.cat([unscaled_targets, unscaled_predictions], dim=2)
    graphs = {i:joint_graphs[ids == i, :, :] for i in set(ids.tolist())}
    for key, g in graphs.items():
        for i, ex in enumerate(g):
            df = pd.DataFrame(ex.numpy(), 
                    index=range(num_horizons - ex.shape[0], num_horizons),
                    columns=['target'] + [f'P{int(q*100)}' for q in config.quantiles])
            fig = df.plot().get_figure()
            ax = fig.get_axes()[0]
            _values = df.values[config.encoder_length-1:,:]
            ax.fill_between(range(num_horizons), _values[:,1], _values[:,-1], alpha=0.2, color='green')
            os.makedirs(os.path.join(args.results, 'single_example_vis', str(key)), exist_ok=True)
            fig.savefig(os.path.join(args.results, 'single_example_vis', str(key), f'{i}.pdf'))

def inference(args, config, model, data_loader, scalers, cat_encodings):
    unscaled_predictions, unscaled_targets, ids, perf_meter = predict(args, config, model, data_loader, scalers, cat_encodings)

    if args.joint_visualization or args.save_predictions:
        ids = torch.from_numpy(ids.squeeze())
        #ids = torch.cat([x['id'][0] for x in data_loader.dataset])
        joint_graphs = torch.cat([unscaled_targets, unscaled_predictions], dim=2)
        graphs = {i:joint_graphs[ids == i, :, :] for i in set(ids.tolist())}
        for key, g in graphs.items(): #timeseries id, joint targets and predictions
            _g = {'targets': g[:,:,0]}
            _g.update({f'P{int(q*100)}':g[:,:,i+1] for i, q in enumerate(config.quantiles)})
            
            if args.joint_visualization:
                summary_writer = SummaryWriter(log_dir=os.path.join(args.results, 'predictions_vis', str(key)))
                for q, t in _g.items(): # target and quantiles, timehorizon values
                    if q == 'targets':
                        targets = torch.cat([t[:,0], t[-1,1:]]) # WIP
                        # We want to plot targets on the same graph as predictions. Probably could be written better.
                        for i, val in enumerate(targets):
                            summary_writer.add_scalars(str(key), {f'{q}':val}, i)
                        continue

                    # Tensor t contains different time horizons which are shifted in phase
                    # Next lines realign them
                    y = t.new_full((t.shape[0] + t.shape[1] -1, t.shape[1]), float('nan'))
                    for i in range(y.shape[1]):
                        y[i:i+t.shape[0], i] = t[:,i]

                    for i, vals in enumerate(y): # timestep, timehorizon values value
                        summary_writer.add_scalars(str(key), {f'{q}_t+{j+1}':v for j,v in enumerate(vals) if v == v}, i)
                summary_writer.close()

            if args.save_predictions:
                for q, t in _g.items():
                    df = pd.DataFrame(t.tolist())
                    df.columns = [f't+{i+1}' for i in range(len(df.columns))]
                    os.makedirs(os.path.join(args.results, 'predictions', str(key)), exist_ok=True)
                    df.to_csv(os.path.join(args.results, 'predictions', str(key), q+'.csv'))

    #losses = QuantileLoss(config)(torch.from_numpy(unscaled_predictions).contiguous(),
    #        torch.from_numpy(unscaled_targets).contiguous()).numpy()
    #normalizer = np.mean(np.abs(unscaled_targets))
    #q_risk = 2 * losses / normalizer
    risk = qrisk(unscaled_predictions, unscaled_targets, np.array(config.quantiles))

    perf_dict = {
                'throughput': perf_meter.avg,
                'latency_avg': perf_meter.total_time/len(perf_meter.intervals),
                'latency_p90': perf_meter.p(90),
                'latency_p95': perf_meter.p(95),
                'latency_p99': perf_meter.p(99),
                'total_infernece_time': perf_meter.total_time,
                }

    return risk, perf_dict


def main(args):
    
    setup_logger(args)
    # Set up model
    state_dict = torch.load(args.checkpoint)
    config = state_dict['config']
    model = TemporalFusionTransformer(config).cuda()
    model.load_state_dict(state_dict['model'])
    model.eval()
    model.cuda()

    # Set up dataset
    test_split = TFTDataset(args.data, config)
    data_loader = DataLoader(test_split, batch_size=args.batch_size, num_workers=4)

    scalers = pickle.load(open(args.tgt_scalers, 'rb'))
    cat_encodings = pickle.load(open(args.cat_encodings, 'rb'))

    if args.visualize:
        # TODO: abstract away all forms of visualization.
        visualize_v2(args, config, model, data_loader, scalers, cat_encodings)

    quantiles, perf_dict = inference(args, config, model, data_loader, scalers, cat_encodings)
    quantiles = {'test_p10': quantiles[0].item(), 'test_p50': quantiles[1].item(), 'test_p90': quantiles[2].item(), 'sum':sum(quantiles).item()}
    finish_log = {**quantiles, **perf_dict}
    dllogger.log(step=(), data=finish_log, verbosity=1)
    print('Test q-risk: P10 {test_p10} | P50 {test_p50} | P90 {test_p90}'.format(**quantiles))
    print('Latency:\n\tAverage {:.3f}s\n\tp90 {:.3f}s\n\tp95 {:.3f}s\n\tp99 {:.3f}s'.format(
        perf_dict['latency_avg'], perf_dict['latency_p90'], perf_dict['latency_p95'], perf_dict['latency_p99']))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        help='Path to the checkpoint')
    parser.add_argument('--data', type=str,
                        help='Path to the test split of the dataset')
    parser.add_argument('--tgt_scalers', type=str,
                        help='Path to the tgt_scalers.bin file produced by the preprocessing')
    parser.add_argument('--cat_encodings', type=str,
                        help='Path to the cat_encodings.bin file produced by the preprocessing')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions - each example on the separate plot')
    parser.add_argument('--joint_visualization', action='store_true', help='Visualize predictions - each timeseries on separate plot. Projections will be concatenated.')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--results', type=str, default='/results')
    parser.add_argument('--log_file', type=str, default='dllogger.json')
    parser.add_argument("--disable_benchmark", action='store_true', help='Disable benchmarking mode')
    ARGS = parser.parse_args()
    main(ARGS)
