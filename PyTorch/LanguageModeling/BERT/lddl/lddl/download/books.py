import argparse
import functools
import multiprocessing
import os
import subprocess
import tqdm

from .utils import download, parse_str_of_num_bytes
from lddl.utils import (expand_outdir_and_mkdir, mkdir,
                        get_all_files_paths_under, attach_bool_arg)


def _get_url():
  return 'https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz'


def attach_args(parser=argparse.ArgumentParser("""
Books Downloader performs the following steps:
- Step 1: Download the compressed bookscorpus from {} into the directory
  specified by the --outdir flag.
- Step 2: Unzip the compressed bookscorpus into raw text files of individual
  books.
- Step 3: Shard the books into text shards in the 'source' subdirectory under
  the directory specified by the --outdir flag. The text shards under the
  `source` subdirectory can then be used as the input to the LDDL preprocessor.
All steps are executed by default. Each step, before it starts, expects the
previous steps already finish. You can turn Step 1 off by --no-download, turn
Step 2 off by --no-unzip, and turn Step 3 off by --no-shard.

Examples:

# Download the compressed bookscorpus into books/books1.tar.gz :
$ download_books --no-unzip --no-shard
$ tree books/  # tree can be installed via `sudo apt install tree`.
books/
└── books1.tar.gz

# Unzip books/books1.tar.gz into individual books:
$ download_books --no-download --no-shard
$ tree books/
books/
├── books1
│   ├── 2020-08-27-epub_urls.txt
│   └── epubtxt
│       ├── 1000-lines-magic-sequence.epub.txt
│       ├── 1000-yards-john-milton-1.epub.txt
│       ...
│       └── zorana-confessions-of-a-small-town-super-villain.epub.txt
├── books1.tar.gz
├── tar.err
└── tar.out

# Shard the books into text shards under books/source which can be read by
# the LDDL preprocessor as input.
$ download_books --no-download --no-unzip
$ tree books/
books/
├── books1
│   ├── 2020-08-27-epub_urls.txt
│   └── epubtxt
│       ├── 1000-lines-magic-sequence.epub.txt
│       ├── 1000-yards-john-milton-1.epub.txt
│       ...
│       └── zorana-confessions-of-a-small-town-super-villain.epub.txt
├── books1.tar.gz
├── source
│   ├── 0.txt
│   ...
│   └── 9.txt
├── tar.err
└── tar.out
# books/source is the input to the LDDL preprocessor.

# Or, we could do all 3 steps together:
$ download_books --outdir books/
""".format(_get_url()))):
  parser.add_argument(
      '--outdir',
      type=str,
      default=None,
      required=True,
      help='Path to the output directory. This directory will be created if not'
      ' already existed.',
  )
  defaults = {
      '--download-chunk-size': 16 * 1024 * 1024,
      '--num-shards': 10,
      '--shard-num-processes': os.cpu_count(),
  }
  attach_bool_arg(
      parser,
      'download',
      default=True,
      help_str='--download is set by default. To skip Step 1, explicitly set '
      '--no-download.',
  )
  attach_bool_arg(
      parser,
      'unzip',
      default=True,
      help_str='--unzip is set by default. To skip Step 2, explicitly set '
      '--no-unzip.',
  )
  attach_bool_arg(
      parser,
      'shard',
      default=True,
      help_str='--shard is set by default. To skip Step 3, explicitly set '
      '--no-shard.',
  )
  parser.add_argument(
      '--download-chunk-size',
      type=functools.partial(parse_str_of_num_bytes, return_str=False),
      default=defaults['--download-chunk-size'],
      metavar="n[KMG]",
      help='The downloading will be performed in a streaming way by looping '
      'over the following steps: (i) transfer a small chunk of data over the '
      'network into the host memory, (ii) write this chunk onto disk. This flag'
      ' indicates the chunk size. Default: {}'.format(
          defaults['--download-chunk-size']),
  )
  parser.add_argument(
      '--num-shards',
      type=int,
      default=defaults['--num-shards'],
      help='The number of text shards into which the books are aggregated. '
      'Default: {}'.format(defaults['--num-shards']),
  )
  parser.add_argument(
      '--shard-num-processes',
      type=int,
      default=defaults['--shard-num-processes'],
      help='The number of processes used to shard all books. '
      'Default: {}'.format(defaults['--shard-num-processes']),
  )
  return parser


def _shard_book(shard):
  shard_path, books = shard
  with open(shard_path, 'w', newline='\n') as shard_file:
    one_line_books = []
    for book in books:
      with open(book, 'r', encoding='utf-8-sig', newline='\n') as book_file:
        book_lines = (bl.strip() for bl in book_file)
        book_lines = [bl for bl in book_lines if len(bl) > 0]
        # The first token is the name of the book.
        book_name = os.path.splitext(os.path.basename(book))[0]
        one_line_books.append(' '.join([book_name] + book_lines))
    shard_file.write('\n'.join(one_line_books))


def _shard_books(books_dir, shards_dir, num_shards, num_processes):
  book_paths = [
      f for f in get_all_files_paths_under(books_dir)
      if os.path.splitext(f)[1] == '.txt'
  ]
  shards = [(
      os.path.join(shards_dir, '{}.txt'.format(shard_idx)),
      book_paths[shard_idx::num_shards],
  ) for shard_idx in range(num_shards)]
  with multiprocessing.Pool(num_processes) as p:
    list(tqdm.tqdm(p.imap(_shard_book, shards), total=len(shards)))


def main(args):
  args.outdir = expand_outdir_and_mkdir(args.outdir)
  target_path = os.path.join(args.outdir, 'books1.tar.gz')
  if args.download:
    download(
        _get_url(),
        target_path,
        chunk_size=args.download_chunk_size,
    )
  if args.unzip:
    print('Unzipping {} ...'.format(target_path))
    out_path = os.path.join(args.outdir, 'tar.out')
    err_path = os.path.join(args.outdir, 'tar.err')
    try:
      subprocess.run(
          ['tar', '-xvzf', target_path, '-C', args.outdir],
          check=True,
          stdout=open(out_path, 'w'),
          stderr=open(err_path, 'w'),
      )
    except subprocess.CalledProcessError as e:
      print(e, 'Please check {} and {}'.format(out_path, err_path))
      raise
  if args.shard:
    books_dir = os.path.join(args.outdir, 'books1', 'epubtxt')
    print('Sharding {} ...'.format(books_dir))
    dask_source_path = os.path.join(args.outdir, 'source')
    mkdir(dask_source_path)
    _shard_books(
        books_dir,
        dask_source_path,
        args.num_shards,
        args.shard_num_processes,
    )
    print('Dask source prepared at {} !'.format(dask_source_path))


def console_script():
  main(attach_args().parse_args())
