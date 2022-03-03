import torch


def barrier():
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.barrier()


def get_rank():
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    rank = torch.distributed.get_rank()
  else:
    rank = 0
  return rank


def get_world_size():
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    world_size = torch.distributed.get_world_size()
  else:
    world_size = 1
  return world_size


def get_nproc_per_node(local_rank):
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    max_local_rank = torch.tensor(
        local_rank,
        device='cuda' if torch.distributed.get_backend() == 'nccl' else 'cpu',
    )
    torch.distributed.all_reduce(
        max_local_rank,
        op=torch.distributed.ReduceOp.MAX,
    )
    nproc_per_node = max_local_rank.item() + 1
  else:
    nproc_per_node = 1
  return nproc_per_node


def get_num_nodes(local_rank=None, nproc_per_node=None):
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    if nproc_per_node is None:
      assert local_rank is not None
      nproc_per_node = get_nproc_per_node(local_rank)
    num_nodes = get_world_size() // nproc_per_node
  else:
    num_nodes = 1
  return num_nodes


def get_node_rank(local_rank=None, nproc_per_node=None):
  """ This assume the training processes are launched via
  torch.distributed.launch.py. Therefore, the ordering scheme of
  rank             -> (node_rank, local_rank) mapping is:
  0                -> (0, 0)
  1                -> (0, 1)
  ...
  nproc_per_node   -> (1, 0)
  nproc_per_node+1 -> (1, 1)
  ...
  """
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    if nproc_per_node is None:
      assert local_rank is not None
      nproc_per_node = get_nproc_per_node(local_rank)
    node_rank = get_rank() // nproc_per_node
  else:
    node_rank = 0
  return node_rank
