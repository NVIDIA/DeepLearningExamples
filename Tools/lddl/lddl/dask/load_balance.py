import argparse
import json
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
import time
from mpi4py import MPI

from lddl.types import File
from lddl.utils import (get_all_files_paths_under, expand_outdir_and_mkdir,
                        get_all_parquets_under, get_all_bin_ids,
                        get_file_paths_for_bin_id, get_num_samples_of_parquet,
                        attach_bool_arg)


class Shard:

  def __init__(self, idx, input_files, outdir, keep_orig=True, postfix=''):
    self.idx = idx
    self._input_files = input_files
    self._outdir = outdir
    self._keep_orig = keep_orig
    self._postfix = postfix

    self._output_file = None

  @property
  def num_samples(self):
    n = 0
    if self._input_files is not None:
      for input_file in self._input_files:
        n += input_file.num_samples
    if self._output_file is not None:
      n += self._output_file.num_samples
    return n

  def __repr__(self):
    return ('Shard(idx={}, input_files={}, outdir={}, keep_orig={}, '
            'postfix={}, output_file={})'.format(
                self.idx,
                self._input_files,
                self._outdir,
                self._keep_orig,
                self._postfix,
                self._output_file,
            ))

  def _read_table(self, path):
    table = pq.read_table(path)
    if not self._keep_orig:  # Only keep the read table in memory.
      os.remove(path)
    return table

  def _read_table_from_file(self, f):
    table = self._read_table(f.path)
    assert f.num_samples == len(table)
    return table

  def _store(self, num_samples, table=None):
    if table is not None:
      assert num_samples == len(table)
    if self._output_file is None:
      self._output_file = File(
          os.path.join(
              self._outdir,
              'shard-{}.parquet{}'.format(self.idx, self._postfix),
          ),
          0,
      )
    else:
      if table is not None:
        table = pa.concat_tables([
            self._read_table_from_file(self._output_file),
            table,
        ])
    self._output_file.num_samples += num_samples
    if table is not None:
      assert self._output_file.num_samples == len(table)
      pq.write_table(table, self._output_file.path)

  def _load(self, num_samples, return_table=False):
    if return_table:
      tables = []
    while num_samples > 0:
      if len(self._input_files) > 0:
        load_file = self._input_files.pop()
      else:
        load_file = self._output_file
        self._output_file = None
      load_num_samples = min(load_file.num_samples, num_samples)
      if return_table:
        load_table = self._read_table_from_file(load_file)
        tables.append(load_table.slice(length=load_num_samples))
      if load_num_samples < load_file.num_samples:
        self._store(
            load_file.num_samples - load_num_samples,
            table=load_table.slice(
                offset=load_num_samples) if return_table else None,
        )
      num_samples -= load_num_samples
    if return_table:
      return pa.concat_tables(tables)

  def balance(larger_shard, smaller_shard, idx):
    assert larger_shard.num_samples > smaller_shard.num_samples
    num_samples_to_transfer = (
        larger_shard.num_samples -
        (larger_shard.num_samples + smaller_shard.num_samples) // 2)
    smaller_shard._store(
        num_samples_to_transfer,
        table=larger_shard._load(
            num_samples_to_transfer,
            return_table=(idx % get_world_size() == get_rank()),
        ),
    )

  def flush(self, idx):
    if idx % get_world_size() == get_rank():
      input_tables = []
    num_samples_to_flush = 0
    while len(self._input_files) > 0:
      input_file = self._input_files.pop()
      num_samples_to_flush += input_file.num_samples
      if idx % get_world_size() == get_rank():
        input_tables.append(self._read_table_from_file(input_file))
    if num_samples_to_flush > 0:
      self._store(
          num_samples_to_flush,
          table=(pa.concat_tables(input_tables) if
                 (idx % get_world_size() == get_rank()) else None),
      )


class Progress:

  def __init__(self, shards):
    num_shards = len(shards)
    total_num_samples = sum((s.num_samples for s in shards))
    base_num_samples_per_shard = total_num_samples // num_shards
    self._targets = {
        base_num_samples_per_shard: num_shards - total_num_samples % num_shards,
        base_num_samples_per_shard + 1: total_num_samples % num_shards,
    }
    self._ready_shards = []

  def __repr__(self):
    s = [
        'Progress(',
        ' Remaining:',
    ]
    s += [
        '  {} shards with {} samples per shard'.format(v, k)
        for k, v in self._targets.items()
    ]
    s += [
        ' Ready:',
        '  {} shards'.format(len(self._ready_shards)),
        ')',
    ]
    return '\n'.join(s)

  def completed(self):
    return sum(self._targets.values()) == 0

  def report(self, shards):
    smaller_shards, larger_shards = [], []
    for shard in shards:
      if shard.num_samples in self._targets:
        self._targets[shard.num_samples] -= 1
        self._ready_shards.append(shard)
        if self._targets[shard.num_samples] == 0:
          del self._targets[shard.num_samples]
      else:
        if shard.num_samples < min(self._targets.keys()):
          smaller_shards.append(shard)
        else:
          larger_shards.append(shard)
    return smaller_shards, larger_shards

  @property
  def ready_shards(self):
    return self._ready_shards


def get_world_size():
  return MPI.COMM_WORLD.Get_size()


def get_rank():
  return MPI.COMM_WORLD.Get_rank()


def barrier():
  return MPI.COMM_WORLD.barrier()


def allreduce(array, op=MPI.SUM):
  MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, array, op=op)


def _build_files(file_paths):
  # Get the number of samples for each file in a collectively distributed
  # approach.
  all_files_num_samples = np.zeros((len(file_paths),), dtype=np.uint64)
  for file_idx in range(get_rank(), len(file_paths), get_world_size()):
    all_files_num_samples[file_idx] = get_num_samples_of_parquet(
        file_paths[file_idx])
  allreduce(all_files_num_samples)
  return sorted(
      [
          File(path, num_samples) for (path, num_samples) in zip(
              file_paths,
              all_files_num_samples.tolist(),
          )
      ],
      key=lambda f: f.num_samples,
  )


def _build_shards(files, num_shards, outdir, keep_orig=True, postfix=''):
  return [
      Shard(
          idx,
          files[idx::num_shards] if idx < len(files) else None,
          outdir,
          keep_orig=keep_orig,
          postfix=postfix,
      ) for idx in range(num_shards)
  ]


def _calculate_mean_std_num_samples(shards):
  num_samples = [shard.num_samples for shard in shards]
  if len(num_samples) > 0:
    return np.mean(num_samples), np.std(num_samples)
  else:
    return np.NAN, np.NAN


def attach_args(parser=argparse.ArgumentParser("""
LDDL Load Balancer for the parquet shards generated by the LDDL Preprocessor

Assume the set of parquet shards generated by the LDDL Preprocessor is P, for
any two parquet shards a and b in P, the LDDL load balancer makes sure that the
numbers of samples in a and b differ *at most* by 1. In other words, the LDDL
load balancer "balances" the number of samples among the parquet shards.

MPI is used to scale the LDDL load balancer to multi-processes and multi-nodes.
MPI can be accessed in various ways. For example, we can access MPI via mpirun:
$ mpirun -c <number of processes per node> --oversubscribe --allow-run-as-root \\
    balance_dask_output ...
We can also access MPI via SLURM in a HPC cluster:
$ srun -l --mpi=pmix --ntasks-per-node=<number of processes per node> \\
    balance_dask_output ...
""")):
  parser.add_argument(
      '--indir',
      type=str,
      required=True,
      help='The path to the directory that contains the parquet shards '
      'generated by the LDDL Preprocessor.',
  )
  parser.add_argument(
      '--outdir',
      type=str,
      default=None,
      help="The path where the balanced parquet shards will be stored. If "
      "unspecified, the balanced parquet shards will be stored in the "
      "directory of '--indir'.",
  )
  parser.add_argument(
      '--num-shards',
      type=int,
      required=True,
      help='The total number of shards that should be balanced into.',
  )
  parser.add_argument(
      '--bin-ids',
      type=int,
      nargs='*',
      default=None,
      help='The bin IDs to perform load balance on (if binning is enabled). If '
      'unspecified, load balance will be performed on all bins.',
  )
  attach_bool_arg(
      parser,
      'keep-orig',
      default=False,
      help_str="If '--keep-orig' is specified, the original unbalanced parquet "
      "shards are kept. By default, those original unbalanced parquet shards "
      "are deleted after the balanced shards are generated.",
  )
  return parser


def _balance(file_paths, num_shards, outdir, keep_orig=True, postfix=''):
  files = _build_files(file_paths)
  shards = _build_shards(
      files,
      num_shards,
      outdir,
      keep_orig=keep_orig,
      postfix=postfix,
  )
  if get_rank() == 0:
    print('Balancing the following {} files into {} shards:'.format(
        len(files), num_shards))
    print('SUM(files.num_samples) = {}, SUM(shards.num_samples) = {}'.format(
        sum((f.num_samples for f in files)),
        sum((s.num_samples for s in shards)),
    ))
  progress = Progress(shards)
  if get_rank() == 0:
    print('Begin with {}'.format(progress))
  iteration = 0
  while not progress.completed():
    smaller_shards, larger_shards = progress.report(shards)
    if get_rank() == 0:
      print('iteration {}, {}, left {}, right {}'.format(
          iteration,
          progress,
          _calculate_mean_std_num_samples(smaller_shards),
          _calculate_mean_std_num_samples(larger_shards),
      ))
    smaller_shards = list(
        sorted(smaller_shards, key=lambda shard: shard.num_samples))
    larger_shards = list(
        sorted(
            larger_shards,
            key=lambda shard: shard.num_samples,
            reverse=True,
        ))
    num_pairs = min(len(smaller_shards), len(larger_shards))
    for i, (smaller_shard, larger_shard) in enumerate(
        zip(smaller_shards[:num_pairs], larger_shards[:num_pairs])):
      larger_shard.balance(smaller_shard, i)
    barrier()
    shards = smaller_shards + larger_shards
    iteration += 1

  [shard.flush(i) for i, shard in enumerate(progress.ready_shards)]
  if get_rank() == 0:
    print('Done!')
  return progress.ready_shards


def _store_num_samples_per_shard(shards, outdir):
  num_samples_per_shard = {
      os.path.basename(shard._output_file.path): shard._output_file.num_samples
      for shard in shards
  }
  with open(os.path.join(outdir, '.num_samples.json'), 'w') as f:
    json.dump(num_samples_per_shard, f)


def main(args):

  if args.outdir is None:
    args.outdir = args.indir
  else:
    args.outdir = expand_outdir_and_mkdir(args.outdir)

  file_paths = get_all_parquets_under(args.indir)
  if args.bin_ids is None:
    bin_ids = get_all_bin_ids(file_paths)
    if len(bin_ids) > 0:
      args.bin_ids = bin_ids
  ready_shards = []
  if args.bin_ids is None:
    if get_rank() == 0:
      print('Load balancing for unbinned files ...')
    ready_shards.extend(
        _balance(file_paths,
                 args.num_shards,
                 args.outdir,
                 keep_orig=args.keep_orig))
  else:
    if get_rank() == 0:
      print('Load balancing for bin_ids = {} ...'.format(args.bin_ids))
    for bin_id in args.bin_ids:
      if get_rank() == 0:
        print('Balancing bin_id = {} ...'.format(bin_id))
      file_paths_current_bin = get_file_paths_for_bin_id(file_paths, bin_id)
      ready_shards.extend(
          _balance(
              file_paths_current_bin,
              args.num_shards,
              args.outdir,
              keep_orig=args.keep_orig,
              postfix='_{}'.format(bin_id),
          ))
  if get_rank() == 0:
    _store_num_samples_per_shard(ready_shards, args.outdir)


def console_script():
  tic = time.perf_counter()
  main(attach_args().parse_args())
  if get_rank() == 0:
    print('Load balancing took {} s!'.format(time.perf_counter() - tic))


def generate_num_samples_cache():
  parser = argparse.ArgumentParser(
      'Generate .num_samples.json for the balanced parquets.')
  parser.add_argument(
      '--indir',
      type=str,
      default=None,
      help='path to the dir that contains the balanced shards',
  )
  args = parser.parse_args()
  file_paths = get_all_parquets_under(args.indir)
  # Get the number of samples for each file in a collectively distributed
  # approach.
  all_files_num_samples = np.zeros((len(file_paths),), dtype=np.uint64)
  for file_idx in range(get_rank(), len(file_paths), get_world_size()):
    all_files_num_samples[file_idx] = get_num_samples_of_parquet(
        file_paths[file_idx])
  allreduce(all_files_num_samples)
  all_files_num_samples = all_files_num_samples.tolist()
  with open(os.path.join(args.indir, '.num_samples.json'), 'w') as nsf:
    json.dump(
        {
            os.path.basename(file_paths[file_idx]):
            all_files_num_samples[file_idx]
            for file_idx in range(len(file_paths))
        },
        nsf,
    )
