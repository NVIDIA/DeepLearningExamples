import argparse
import functools
import multiprocessing
import os
import subprocess
import tqdm
import xml.etree.ElementTree as ET

from .utils import download, parse_str_of_num_bytes
from lddl.utils import (expand_outdir_and_mkdir, mkdir,
                        get_all_files_paths_under, attach_bool_arg)


def _get_url(lang):
  assert lang in {'en', 'zh'}
  return ('https://dumps.wikimedia.org/{lang}wiki/latest'
          '/{lang}wiki-latest-pages-articles.xml.bz2'.format(lang=lang))


def _get_download_target_filename(lang):
  return 'wikicorpus-{}.xml.bz2'.format(lang)


def _prepare_one_shard(shard):
  source_shard_path, extract_shard_path = shard
  articles = []
  with open(extract_shard_path, 'r', newline='\n') as extract_shard_file:
    article_open = None
    article_lines = []
    for line in extract_shard_file:
      if '<doc id=' in line:
        article_open = line
      elif '</doc>' in line:
        article_id = 'wiki-' + ET.fromstring(article_open + line).attrib['id']
        article_open = None
        # article_lines[0] is the title
        if len(article_lines) > 1:
          # The first token is the article id.
          articles.append(' '.join([article_id] + article_lines[1:]))
        article_lines = []
      else:
        if article_open:
          line = line.strip()
          if len(line) > 0:
            article_lines.append(line.strip())

  if len(articles) > 0:
    print('{} -> {}'.format(extract_shard_path, source_shard_path))
    with open(source_shard_path, 'w', newline='\n') as source_shard_file:
      source_shard_file.write('\n'.join(articles))


def _prepare_dask_source(extract_path, dask_source_path, num_processes):
  extracted_shards_paths = [
      p for p in get_all_files_paths_under(extract_path) if 'wiki_' in p
  ]
  shards = [(os.path.join(dask_source_path, '{}.txt'.format(i)), esp)
            for i, esp in enumerate(extracted_shards_paths)]

  with multiprocessing.Pool(num_processes) as p:
    list(tqdm.tqdm(p.imap(_prepare_one_shard, shards), total=len(shards)))


def _download_and_extract(
    lang='en',
    to_download=True,
    to_extract=True,
    to_prepare_source=True,
    download_chunk_size=16 * 1024 * 1024,
    extract_shard_size='128M',
    outdir=None,
    num_processes=os.cpu_count(),
):
  if lang not in {'en', 'zh'}:
    raise ValueError('Language {} not supported!'.format(lang))

  url = _get_url(lang)
  target_filename = _get_download_target_filename(lang)
  target_path = os.path.join(outdir, target_filename)

  if to_download:
    download(url, target_path, chunk_size=download_chunk_size)

  extract_path = os.path.join(outdir, 'extracted', lang)
  if to_extract:
    mkdir(extract_path)
    print('Extracting {} ...'.format(target_path))
    subprocess.run(
        [
            'python',
            '-m',
            'wikiextractor.WikiExtractor',
            target_path,
            '--output',
            extract_path,
            '--bytes',
            extract_shard_size,
            '--processes',
            str(num_processes),
        ],
        check=True,
        stdout=open(os.path.join(extract_path, 'WikiExtractor.out'), 'w'),
        stderr=open(os.path.join(extract_path, 'WikiExtractor.err'), 'w'),
    )

  if to_prepare_source:
    print('Preparing dask source from {} ...'.format(extract_path))
    dask_source_path = os.path.join(outdir, 'source', lang)
    mkdir(dask_source_path)
    _prepare_dask_source(extract_path, dask_source_path, num_processes)
    print('Dask source prepared at {} !'.format(dask_source_path))


def attach_args(parser=argparse.ArgumentParser("""
Wikipedia Downloader performs the following steps:
- Step 1: Download the Wikipedia dumps from {} into the directory specified by
  the --outdir flag.
- Step 2: Extract the raw text from the Wikipedia dumps which are originally in
  the XML format.
- Step 3: Prepare and aggregate the raw text into text shards in the 'source'
  subdirectory under the directory specified by the --outdir flag. The text
  shards under the 'source' subdirectory can then be used as the input to the
  LDDL preprocessor.
All steps are executed by default. Each step, before it starts, expects the
previous steps already finish. You can turn Step 1 off by --no-download, turn
Step 2 off by --no-extract, and turn Step 3 off by --no-prepare-source.

Examples:

# Download the English Wikipedia dumps into wikipedia/wikicorpus-en.xml.bz2 :
$ download_wikipedia --outdir wikipedia/ --no-extract --no-prepare-source
$ tree wikipedia/  # tree can be installed via `sudo apt install tree`.
wikipedia/
└── wikicorpus-en.xml.bz2

# Extract the raw text from the English Wikipedia dumps:
$ download_wikipedia --outdir wikipedia/ --no-download --no-prepare-source
$ tree wikipedia/
wikipedia/
├── extracted
│   └── en
│       ├── AA
│       │   ├── wiki_00
│       │   ├── wiki_01
│       │   ...
│       │   └── wiki_30
│       ├── WikiExtractor.err
│       └── WikiExtractor.out
└── wikicorpus-en.xml.bz2

# Prepare and aggregate the raw text into text shards under wikipedia/source
# which can be read by the LDDL preprocessor as input:
$ download_wikipedia --outdir wikipedia/ --no-download --no-extract
$ tree wikipedia/
wikipedia/
├── extracted
│   └── en
│       ├── AA
│       │   ├── wiki_00
│       │   ├── wiki_01
│       │   ...
│       │   └── wiki_30
│       ├── WikiExtractor.err
│       └── WikiExtractor.out
├── source
│   └── en
│       ├── 0.txt
│       ├── 1.txt
│       ...
│       └── 30.txt
└── wikicorpus-en.xml.bz2
# wikipedia/source/ is the input to the LDDL preprocessor.

# Or, we could do all 3 steps together:
$ download_wikipedia --outdir wikipedia/
""".format(_get_url('en')))):
  parser.add_argument(
      '--outdir',
      type=str,
      default=None,
      required=True,
      help='Path to the output directory. This directory will be created if not'
      ' already existed.',
  )
  defaults = {
      '--langs': ['en'],
      '--download-chunk-size': 16 * 1024 * 1024,
      '--extract-shard-size': '512M',
      '--num-processes': os.cpu_count(),
  }
  parser.add_argument(
      '--langs',
      default=defaults['--langs'],
      nargs='+',
      choices=['en', 'zh'],
      help='Language of the wikipedia dumps to download. Default: {}'.format(
          defaults['--langs']),
  )
  attach_bool_arg(
      parser,
      'download',
      default=True,
      help_str='--download is set by default. To skip Step 1, explicitly set '
      '--no-download.',
  )
  attach_bool_arg(
      parser,
      'extract',
      default=True,
      help_str='--extract is set by default. To skip Step 2, explicitly set '
      '--no-extract.')
  attach_bool_arg(
      parser,
      'prepare-source',
      default=True,
      help_str='--prepare-source is set by default. To skip Step 3, explicitly '
      'set --no-prepare-source.',
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
      '--extract-shard-size',
      type=functools.partial(parse_str_of_num_bytes, return_str=True),
      default=defaults['--extract-shard-size'],
      metavar="n[KMG]",
      help='The size of each text shard. Default: {}'.format(
          defaults['--extract-shard-size']),
  )
  parser.add_argument(
      '--num-processes',
      type=int,
      default=os.cpu_count(),
      help='Num of processes to use. Default: {}'.format(
          defaults['--num-processes']),
  )
  return parser


def main(args):
  args.outdir = expand_outdir_and_mkdir(args.outdir)
  for lang in args.langs:
    _download_and_extract(
        lang=lang,
        to_download=args.download,
        to_extract=args.extract,
        to_prepare_source=args.prepare_source,
        download_chunk_size=args.download_chunk_size,
        extract_shard_size=args.extract_shard_size,
        outdir=args.outdir,
        num_processes=args.num_processes,
    )


def console_script():
  main(attach_args().parse_args())
