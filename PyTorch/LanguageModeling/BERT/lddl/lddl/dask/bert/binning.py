from dask.highlevelgraph import HighLevelGraph

# to_dataframe
from dask.base import tokenize
from dask.bag.core import reify
import itertools

# to_parquet
import tlz as toolz
from fsspec.core import get_fs_token_paths
from dask.base import compute_as_if_collection
from dask.delayed import Delayed
from dask.utils import apply
from dask.dataframe.core import Scalar
from dask.dataframe.io.parquet.core import get_engine
from dask.dataframe.io.parquet.arrow import _index_in_schema
import pyarrow.parquet as pq
try:
  import snappy

  snappy.compress
except (ImportError, AttributeError):
  snappy = None

NONE_LABEL = "__null_dask_index__"

# to_textfiles
import io
import uuid
from dask.bytes import open_files
from dask.utils import ensure_unicode, ensure_bytes, system_encoding
from contextlib import ExitStack

#
# dataframes
#


def _to_dataframe_binned(seq, columns, dtypes, bin_size, nbins):
  import pandas as pd

  seq = reify(seq)
  if not isinstance(seq, list):
    seq = list(seq)

  seqs = [[] for _ in range(nbins)]
  for i, iseq in enumerate(seq):
    seq_len = iseq['num_tokens']
    bin_id = (seq_len - 1) // bin_size
    bin_id = nbins - 1 if bin_id > nbins - 1 else bin_id
    seqs[bin_id].append(iseq)

  dfl = list(
      map(
          lambda l: pd.DataFrame(
              l,
              columns=list(columns),
          ).astype(dtypes, copy=False),
          seqs,
      ))

  dfs = pd.concat(dfl, keys=list(map(str, list(range(nbins)))))

  # Add a bin_id column
  dfs['bin_id'] = list(
      itertools.chain.from_iterable(
          [[i] * len(bingrp) for i, bingrp in enumerate(seqs)]))

  return dfs


def to_dataframe_binned(self, bin_size, nbins, meta=None, columns=None):
  import pandas as pd
  import dask.dataframe as dd

  if meta is None:
    head = self.take(1, warn=False)
    if len(head) == 0:
      raise ValueError("`dask.bag.Bag.to_dataframe` failed to "
                       "properly infer metadata, please pass in "
                       "metadata via the `meta` keyword")
      meta_nobin = pd.DataFrame(list(head), columns=columns)
  elif columns is not None:
    raise ValueError("Can't specify both `meta` and `columns`")
  else:
    meta_nobin = dd.utils.make_meta(meta, parent_meta=pd.DataFrame())
  # Serializing the columns and dtypes is much smaller than serializing
  # the empty frame
  cols = list(meta_nobin.columns)
  dtypes = meta_nobin.dtypes.to_dict()
  name = "to_dataframe-binned-" + tokenize(self, cols, dtypes)
  dsk = self.__dask_optimize__(self.dask, self.__dask_keys__())

  for i in range(self.npartitions):
    dsk[(name, i)] = (_to_dataframe_binned, (self.name, i), cols, dtypes,
                      bin_size, nbins)

  # Update the meta
  meta['bin_id'] = int
  meta = dd.utils.make_meta(meta, parent_meta=pd.DataFrame())

  divisions = [None] * (self.npartitions + 1)
  return dd.DataFrame(dsk, name, meta, divisions)


#
# parquet files
#


def to_parquet_binned(
    df,
    path,
    nbins,
    engine="auto",
    compression="default",
    write_index=True,
    append=False,
    overwrite=False,
    ignore_divisions=False,
    partition_on=None,
    storage_options=None,
    custom_metadata=None,
    write_metadata_file=True,
    compute=True,
    compute_kwargs=None,
    schema=None,
    **kwargs,
):
  compute_kwargs = compute_kwargs or {}

  if compression == "default":
    if snappy is not None:
      compression = "snappy"
    else:
      compression = None

  partition_on = partition_on or []
  if isinstance(partition_on, str):
    partition_on = [partition_on]

  if set(partition_on) - set(df.columns):
    raise ValueError("Partitioning on non-existent column. "
                     "partition_on=%s ."
                     "columns=%s" % (str(partition_on), str(list(df.columns))))

  if isinstance(engine, str):
    engine = get_engine(engine)

  if hasattr(path, "name"):
    path = stringify_path(path)
  fs, _, _ = get_fs_token_paths(path,
                                mode="wb",
                                storage_options=storage_options)
  # Trim any protocol information from the path before forwarding
  path = fs._strip_protocol(path)

  if overwrite:
    if isinstance(fs, LocalFileSystem):
      working_dir = fs.expand_path(".")[0]
      if path.rstrip("/") == working_dir.rstrip("/"):
        raise ValueError(
            "Cannot clear the contents of the current working directory!")
    if append:
      raise ValueError("Cannot use both `overwrite=True` and `append=True`!")
    if fs.exists(path) and fs.isdir(path):
      # Only remove path contents if
      # (1) The path exists
      # (2) The path is a directory
      # (3) The path is not the current working directory
      fs.rm(path, recursive=True)

  # Save divisions and corresponding index name. This is necessary,
  # because we may be resetting the index to write the file
  division_info = {"divisions": df.divisions, "name": df.index.name}
  if division_info["name"] is None:
    # As of 0.24.2, pandas will rename an index with name=None
    # when df.reset_index() is called.  The default name is "index",
    # but dask will always change the name to the NONE_LABEL constant
    if NONE_LABEL not in df.columns:
      division_info["name"] = NONE_LABEL
    elif write_index:
      raise ValueError(
          "Index must have a name if __null_dask_index__ is a column.")
    else:
      warnings.warn("If read back by Dask, column named __null_dask_index__ "
                    "will be set to the index (and renamed to None).")

  # There are some "resrved" names that may be used as the default column
  # name after resetting the index. However, we don't want to treat it as
  # a "special" name if the string is already used as a "real" column name.
  reserved_names = []
  for name in ["index", "level_0"]:
    if name not in df.columns:
      reserved_names.append(name)

  # If write_index==True (default), reset the index and record the
  # name of the original index in `index_cols` (we will set the name
  # to the NONE_LABEL constant if it is originally `None`).
  # `fastparquet` will use `index_cols` to specify the index column(s)
  # in the metadata.  `pyarrow` will revert the `reset_index` call
  # below if `index_cols` is populated (because pyarrow will want to handle
  # index preservation itself).  For both engines, the column index
  # will be written to "pandas metadata" if write_index=True
  index_cols = []
  if write_index:
    real_cols = set(df.columns)
    none_index = list(df._meta.index.names) == [None]
    df = df.reset_index()
    if none_index:
      df.columns = [
          c if c not in reserved_names else NONE_LABEL for c in df.columns
      ]
    index_cols = [c for c in set(df.columns) - real_cols]
  else:
    # Not writing index - might as well drop it
    df = df.reset_index(drop=True)

  _to_parquet_kwargs = {
      "engine",
      "compression",
      "write_index",
      "append",
      "ignore_divisions",
      "partition_on",
      "storage_options",
      "write_metadata_file",
      "compute",
  }
  kwargs_pass = {k: v for k, v in kwargs.items() if k not in _to_parquet_kwargs}

  # Engine-specific initialization steps to write the dataset.
  # Possibly create parquet metadata, and load existing stuff if appending
  meta, schema, i_offset = engine.initialize_write(
      df,
      fs,
      path,
      append=append,
      ignore_divisions=ignore_divisions,
      partition_on=partition_on,
      division_info=division_info,
      index_cols=index_cols,
      schema=schema,
      **kwargs_pass,
  )

  # Use i_offset and df.npartitions to define file-name list
  filenames = [
      "part.%i.parquet" % (i + i_offset) for i in range(df.npartitions)
  ]

  # Construct IO graph
  dsk = {}
  name = "to-parquet-binned" + tokenize(
      df,
      fs,
      path,
      append,
      ignore_divisions,
      partition_on,
      division_info,
      index_cols,
      schema,
  )
  part_tasks = []
  kwargs_pass["fmd"] = meta
  kwargs_pass["compression"] = compression
  kwargs_pass["index_cols"] = index_cols
  kwargs_pass["schema"] = schema
  if custom_metadata:
    if b"pandas" in custom_metadata.keys():
      raise ValueError(
          "User-defined key/value metadata (custom_metadata) can not "
          "contain a b'pandas' key.  This key is reserved by Pandas, "
          "and overwriting the corresponding value can render the "
          "entire dataset unreadable.")
    kwargs_pass["custom_metadata"] = custom_metadata
  # Override write_partition to write binned parquet files
  engine.write_partition = write_partition_binned
  for d, filename in enumerate(filenames):
    dsk[(name, d)] = (
        apply,
        engine.write_partition,
        [
            engine,
            (df._name, d),
            path,
            fs,
            filename,
            partition_on,
            write_metadata_file,
            nbins,
        ],
        toolz.merge(kwargs_pass, {"head": True}) if d == 0 else kwargs_pass,
    )
    part_tasks.append((name, d))

  final_name = "metadata-" + name
  # Collect metadata and write _metadata

  if write_metadata_file:
    dsk[(final_name, 0)] = (
        apply,
        engine.write_metadata,
        [
            part_tasks,
            meta,
            fs,
            path,
        ],
        {
            "append": append,
            "compression": compression
        },
    )
  else:
    dsk[(final_name, 0)] = (lambda x: None, part_tasks)

  graph = HighLevelGraph.from_collections(final_name, dsk, dependencies=[df])
  out = Delayed(name, graph)

  if compute:
    return compute_as_if_collection(Scalar, graph, [(final_name, 0)],
                                    **compute_kwargs)
  else:
    return Scalar(graph, final_name, "")


def write_partition_binned(
    cls,
    df,
    path,
    fs,
    filename,
    partition_on,
    return_metadata,
    nbins,
    fmd=None,
    compression=None,
    index_cols=None,
    schema=None,
    head=False,
    custom_metadata=None,
    **kwargs,
):
  _meta = None
  preserve_index = False
  if _index_in_schema(index_cols, schema):
    df.set_index(index_cols, inplace=True)
    preserve_index = True
  else:
    index_cols = []

  for ibin in range(nbins):

    dff = df[df.bin_id == ibin]

    filename_b = "%s_%d" % (filename, ibin)

    t = cls._pandas_to_arrow_table(
        dff,
        preserve_index=preserve_index,
        schema=schema,
    )
    if custom_metadata:
      _md = t.schema.metadata
      _md.update(custom_metadata)
      t = t.replace_schema_metadata(metadata=_md)

    if partition_on:
      md_list = _write_partitioned(
          t,
          path,
          filename_b,
          partition_on,
          fs,
          index_cols=index_cols,
          compression=compression,
          **kwargs,
      )
      if md_list:
        _meta = md_list[0]
        for i in range(1, len(md_list)):
          _append_row_groups(_meta, md_list[i])
    else:
      md_list = []
      with fs.open(fs.sep.join([path, filename_b]), "wb") as fil:
        pq.write_table(
            t,
            fil,
            compression=compression,
            metadata_collector=md_list,
            **kwargs,
        )
      if md_list:
        _meta = md_list[0]
        _meta.set_file_path(filename)

  # Return the schema needed to write the metadata
  if return_metadata:
    d = {"meta": _meta}
    if head:
      # Only return schema if this is the "head" partition
      d["schema"] = t.schema
    return [d]
  else:
    return []


#
# text files
#


class file_namer(object):

  def __init__(self, bin_size, nbins, prefix=""):
    self.__bin_size = bin_size
    self.__nbins = nbins
    self.__prefix = prefix

  def name_function(self, i):
    num = i // self.__nbins
    bin_val = i % self.__nbins
    return '%s%d_%d' % (self.__prefix, num, bin_val)


def _to_textfiles_chunk_binned(data, lazy_files, last_endline, bin_size):
  nbins = len(lazy_files)
  with ExitStack() as stack:
    fs = [stack.enter_context(lazy_file) for lazy_file in lazy_files]
    if isinstance(fs[0], io.TextIOWrapper):
      endline = "\n"
      ensure = ensure_unicode
    else:
      endline = b"\n"
      ensure = ensure_bytes
    starteds = [False] * nbins
    for d in data:
      # Assuming the last character containes the number of tokens.
      seq_len = int(d.split()[-1])
      bin_id = (seq_len - 1) // bin_size
      bin_id = nbins - 1 if bin_id > nbins - 1 else bin_id
      if starteds[bin_id]:
        fs[bin_id].write(endline)
      else:
        starteds[bin_id] = True
      fs[bin_id].write(ensure(d))
    if last_endline:
      for f in fs:
        f.write(endline)


def to_textfiles_binned(b,
                        path,
                        bin_size=64,
                        nbins=8,
                        compression="infer",
                        encoding=system_encoding,
                        compute=True,
                        storage_options=None,
                        last_endline=False,
                        **kwargs):

  mode = "wb" if encoding is None else "wt"
  files = open_files(path,
                     compression=compression,
                     mode=mode,
                     encoding=encoding,
                     name_function=file_namer(bin_size, nbins).name_function,
                     num=b.npartitions * nbins,
                     **(storage_options or {}))

  name = "to-textfiles-binned-" + uuid.uuid4().hex
  dsk = {(name, i): (_to_textfiles_chunk_binned, (b.name, i),
                     files[k:k + nbins], last_endline, bin_size)
         for i, k in enumerate(range(0, len(files), nbins))}
  graph = HighLevelGraph.from_collections(name, dsk, dependencies=[b])
  out = type(b)(graph, name, b.npartitions)

  if compute:
    out.compute(**kwargs)
    return [f.path for f in files]
  else:
    return out.to_delayed()
