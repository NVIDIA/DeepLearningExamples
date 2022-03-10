import argparse
import datetime
import functools
import logging
import multiprocessing
import os
import socket
import threading
import time
import tqdm
from newsplease.crawler import commoncrawl_crawler

from lddl.utils import (expand_outdir_and_mkdir, mkdir,
                        get_all_files_paths_under, attach_bool_arg)


def attach_args(parser=argparse.ArgumentParser("""
Common Crawl Downloader performs the following steps:
- Step 1: Download the commoncrawl.org's news archive using newsplease
  (https://github.com/fhamborg/news-please) and extract the raw text of the
  articles to the directory specified by the --outdir flag.
- Step 2: Prepare and aggregate the raw text into text shards in the 'source'
  subdirectory under the directory specified by the --outdir flag. The text
  shards under the 'source' subdirectory can then be used as the input to the
  LDDL preprocessor.
All steps are executed by default. Each step, before it starts, expects the
previous steps already finish. You can turn Step 1 off by --no-newsplease, and
turn Step 2 off by --no-shard.

Examples:

# Download the Common Crawl news archive from .warc files released in Oct. 2021
# and extract the news articles that are published from Jan. 3rd, 2000 to Mar.
# 1st, 2010 to common_crawl/txt/ :
$ download_common_crawl \
    --outdir common_crawl/ \
    --no-shard \
    --warc-files-start-date 2021-10-01 \
    --warc-files-end-date 2021-11-01 \
    --start-date 2000-01-03 \
    --end-date 2010-03-01
$ tree -L 1 common_crawl/  # tree can be installed via `sudo apt install tree`
common_crawl/
├── txt
└── warc

# Shard the news articles into text shards under common_crawl/source which can
# be read by the LDDL preprocessor as input:
$ download_common_crawl --outdir common_crawl/ --no-newsplease
$ tree -L 1 common_crawl/
common_crawl/
├── source
├── txt
└── warc
# common_crawl/source is the input to the LDDL preprocessor.

# Or, we could do all 2 steps together:
$ download_common_crawl \
    --outdir common_crawl/ \
    --warc-files-start-date 2021-10-01 \
    --warc-files-end-date 2021-11-01 \
    --start-date 2000-01-03 \
    --end-date 2010-03-01
""")):
  parser.add_argument(
      '--outdir',
      type=str,
      default=None,
      required=True,
      help='Path to the output directory. This directory will be created if not'
      ' already existed.',
  )
  defaults = {
      '--prefix': socket.gethostname(),
      '--number-of-extraction-processes': os.cpu_count(),
      '--valid-hosts': [],
      '--start-date': None,
      '--start-date-format': '%Y-%m-%d',
      '--end-date': None,
      '--end-date-format': '%Y-%m-%d',
      '--warc-files-start-date': None,
      '--warc-files-start-date-format': '%Y-%m-%d',
      '--warc-files-end-date': None,
      '--warc-files-end-date-format': '%Y-%m-%d',
      '--articles-per-write': 1024,
      '--langs': ['en'],
      '--num-shards': 8,
      '--number-of-sharding-processes': os.cpu_count(),
  }
  parser.add_argument(
      '--prefix',
      type=str,
      default=defaults['--prefix'],
      help='A prefix string that is included in the article ID and output file '
      'name of the raw text. The is useful when you need to distribute Step 1 '
      'to many nodes, then merge the downloaded raw text onto a single node to '
      'perform Step 2. Default: {}'.format(defaults['--prefix']),
  )
  parser.add_argument(
      '--number-of-extraction-processes',
      type=int,
      default=defaults['--number-of-extraction-processes'],
      help='The number of processes used for raw text extraction by newsplease.'
      ' Default: {}'.format(defaults['--number-of-extraction-processes']),
  )
  parser.add_argument(
      '--valid-hosts',
      type=str,
      nargs='*',
      default=defaults['--valid-hosts'],
      help='Only news articles from the hosts in this list are kept. '
      'Default: {} (any host is OK); example: [\'elrancaguino.cl\']'.format(
          defaults['--valid-hosts']),
  )
  parser.add_argument(
      '--start-date',
      type=str,
      default=defaults['--start-date'],
      help='Only news articles published after this start date are kept. '
      'Default: {} (any date is OK as start date)'.format(
          defaults['--start-date']),
  )
  parser.add_argument(
      '--start-date-format',
      type=str,
      default=defaults['--start-date-format'],
      help='The datetime format of the start date specified by --start-date. '
      'Please refer to '
      'https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes'
      ' for details of the datetime format string. Default: %%Y-%%m-%%d',
  )
  parser.add_argument(
      '--end-date',
      type=str,
      default=defaults['--end-date'],
      help='Only news articles published before this end date are kept. '
      'Default: {} (any date is OK as end date)'.format(defaults['--end-date']),
  )
  parser.add_argument(
      '--end-date-format',
      type=str,
      default=defaults['--end-date-format'],
      help='The datetime format of the end date specified by --end-date. Please'
      ' refer to '
      'https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes'
      ' for details of the datetime format string. Default: %%Y-%%m-%%d',
  )
  parser.add_argument(
      '--warc-files-start-date',
      type=str,
      default=defaults['--warc-files-start-date'],
      help='Only .warc files published after this start date are downloaded. '
      'Therefore, you can use this flag to control how much data you want to '
      'download. Default: {} (the date when Common Crawl founded)'.format(
          defaults['--warc-files-start-date']),
  )
  parser.add_argument(
      '--warc-files-start-date-format',
      type=str,
      default=defaults['--warc-files-start-date-format'],
      help='The datetime format of the start date specified by '
      '--warc-files-start-date. Pleas refer to '
      'https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes'
      ' for details of the datetime format string. Default: %%Y-%%m-%%d',
  )
  parser.add_argument(
      '--warc-files-end-date',
      type=str,
      default=defaults['--warc-files-end-date'],
      help='Only .warc files published before this end date are downloaded. '
      'Therefore, you can use this flag to control how much data you want to '
      'download. Default: {} (today)'.format(defaults['--warc-files-end-date']),
  )
  parser.add_argument(
      '--warc-files-end-date-format',
      type=str,
      default=defaults['--warc-files-end-date-format'],
      help='The datetime format of the end date specified by '
      '--warc-files-end-date. Pleas refer to '
      'https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes'
      ' for details of the datetime format string. Default: %%Y-%%m-%%d',
  )
  attach_bool_arg(
      parser,
      'strict-date',
      default=True,
      help_str='If date filtering is strict and newsplease could not detect '
      'the published date of an article, the article will be discarded. '
      '--strict-date is set by default. To turn this off, explicitly set '
      '--no-strict-date.',
  )
  attach_bool_arg(
      parser,
      'reuse-previously-downloaded-files',
      default=True,
      help_str='If reusing previously downloaded files, the script checks '
      'whether a file has been downloaded already and uses that file instead of'
      ' downloading again. Note that there is no check whether the file has '
      'been downloaded completely or is valid! '
      '--reuse-previously-downloaded-files is set by default. To turn this off,'
      ' explicitly set --no-reuse-previously-downloaded-files.',
  )
  attach_bool_arg(
      parser,
      'continue-after-error',
      default=True,
      help_str='If this flag is set, downloading will continue even after '
      'newsplease encounters an error. --continue-after-error is set by '
      'default. To turn this off, explicitly set --no-continue-after-error.',
  )
  attach_bool_arg(
      parser,
      'show-download-progress',
      default=False,
      help_str='If this flag is set, show the progress of downloading the WARC '
      'files. --show-download-progress is NOT set by default.',
  )
  attach_bool_arg(
      parser,
      'delete-warc-after-extraction',
      default=True,
      help_str='If this flag is set, the WARC file will be deleted after all '
      'articles have been extracted from it. --delete-warc-after-extraction is'
      ' set by default. To turn this off, explicitly set '
      '--no-delete-warc-after-extraction.',
  )
  attach_bool_arg(
      parser,
      'continue-process',
      default=True,
      help_str='If this flag is set, newsplease will continue extraction from '
      'the latest fully downloaded but not fully extracted WARC files and then '
      'crawling new WARC files. This assumes that the filter criteria have not '
      'been changed since the previous run! --continue-process is set by '
      'default. To turn this off, explicitly set --no-continue-process.',
  )
  parser.add_argument(
      '--articles-per-write',
      type=int,
      default=defaults['--articles-per-write'],
      help='The articles will be extracted in a streaming way by looping the '
      'following steps: (i) download and extract a small number of articles, '
      '(ii) write this small number of articles to disk. This flag indicates '
      'how many articles cached in memory before a flushing write. Default: {}'.
      format(defaults['--articles-per-write']),
  )
  parser.add_argument(
      '--langs',
      default=defaults['--langs'],
      nargs='+',
      choices=['en'],
      help='Only news articles written in the languages in this list are kept. '
      'Default: {}'.format(defaults['--langs']),
  )
  attach_bool_arg(
      parser,
      'newsplease',
      default=True,
      help_str='--newsplease is set by default. To skip Step 1, explicitly set'
      ' --no-newsplease.',
  )
  attach_bool_arg(
      parser,
      'shard',
      default=True,
      help_str='--shard is set by default. To skip Step 2, explicitly set '
      '--no-shard.',
  )
  parser.add_argument(
      '--num-shards',
      type=int,
      default=defaults['--num-shards'],
      help='The number of text shards into which news articles are aggregated. '
      'Default: {}'.format(defaults['--num-shards']),
  )
  parser.add_argument(
      '--number-of-sharding-processes',
      type=int,
      default=defaults['--number-of-sharding-processes'],
      help='The number of processes used to shard all news articles. '
      'Default: {}'.format(defaults['--number-of-sharding-processes']),
  )
  return parser


class ThreadLocal(threading.local):

  def __init__(self):
    self.out_file = None
    self.articles = []
    self.articles_count = 0
    self.warcs_count = 0


_thread_local = ThreadLocal()


def _flatten(s):
  return ' '.join((l for l in s.splitlines()))


def _flush_articles(
    out_file,
    articles,
    txt_dir,
    warcs_count,
    prefix=socket.gethostname(),
):
  if out_file is None:
    out_file = open(
        os.path.join(
            txt_dir,
            '{}-{}-{}-{}-{}.txt'.format(
                prefix,
                os.getpid(),
                threading.get_ident(),
                warcs_count,
                time.time_ns(),
            ),
        ),
        'w',
    )
    print('{} opened for writing!'.format(out_file.name))
  else:
    out_file.write('\n')
  out_file.write('\n'.join(articles))
  articles.clear()
  return out_file


def _on_valid_article_extracted(
    article,
    articles_per_write=None,
    langs=None,
    txt_dir=None,
    prefix=socket.gethostname(),
):
  if article.language in langs and article.maintext is not None:
    _thread_local.articles.append('{}-{}-{}-{}-{}'.format(
        prefix,
        os.getpid(),
        threading.get_ident(),
        _thread_local.articles_count,
        time.time_ns(),
    ) + ' ' + _flatten(article.maintext))
    _thread_local.articles_count += 1
  if len(_thread_local.articles) > articles_per_write:
    _thread_local.out_file = _flush_articles(
        _thread_local.out_file,
        _thread_local.articles,
        txt_dir,
        _thread_local.warcs_count,
        prefix=prefix,
    )


def _on_warc_completed(
    warc_path,
    counter_article_passed,
    counter_article_discarded,
    counter_article_error,
    counter_article_total,
    counter_warc_processed,
    txt_dir=None,
    prefix=socket.gethostname(),
):
  if len(_thread_local.articles) > 0:
    _thread_local.out_file = _flush_articles(
        _thread_local.out_file,
        _thread_local.articles,
        txt_dir,
        _thread_local.warcs_count,
        prefix=prefix,
    )
  if _thread_local.out_file is not None:
    print('Closing {} !'.format(_thread_local.out_file.name))
    _thread_local.out_file.close()
    _thread_local.out_file = None
  _thread_local.warcs_count += 1


def _aggregate_news(shard):
  shard_path, news_paths = shard
  with open(shard_path, 'w', newline='\n') as shard_file:
    for i, news_path in enumerate(news_paths):
      if i > 0:
        shard_file.write('\n')
      with open(news_path, 'r', newline='\n') as news_file:
        shard_file.write(news_file.read())


def _shard_news(txt_dir, source_dir, num_shards, num_processes):
  news_paths = [
      f for f in get_all_files_paths_under(txt_dir)
      if os.path.splitext(f)[1] == '.txt'
  ]
  shards = [(
      os.path.join(source_dir, '{}.txt'.format(shard_idx)),
      news_paths[shard_idx::num_shards],
  ) for shard_idx in range(num_shards)]
  with multiprocessing.Pool(num_processes) as p:
    list(tqdm.tqdm(p.imap(_aggregate_news, shards), total=len(shards)))


def main(args):
  if args.start_date is not None:
    args.start_date = datetime.datetime.strptime(
        args.start_date,
        args.start_date_format,
    )
  if args.end_date is not None:
    args.end_date = datetime.datetime.strptime(
        args.end_date,
        args.end_date_format,
    )
  if args.warc_files_start_date is not None:
    args.warc_files_start_date = datetime.datetime.strptime(
        args.warc_files_start_date,
        args.warc_files_start_date_format,
    )
  if args.warc_files_end_date is not None:
    args.warc_files_end_date = datetime.datetime.strptime(
        args.warc_files_end_date,
        args.warc_files_end_date_format,
    )
  args.outdir = expand_outdir_and_mkdir(args.outdir)
  txt_dir = os.path.join(args.outdir, 'txt')
  if args.newsplease:
    mkdir(txt_dir)
    commoncrawl_crawler.crawl_from_commoncrawl(
        functools.partial(
            _on_valid_article_extracted,
            articles_per_write=args.articles_per_write,
            langs=set(args.langs),
            txt_dir=txt_dir,
            prefix=args.prefix,
        ),
        callback_on_warc_completed=functools.partial(
            _on_warc_completed,
            txt_dir=txt_dir,
            prefix=args.prefix,
        ),
        valid_hosts=args.valid_hosts,
        start_date=args.start_date,
        end_date=args.end_date,
        warc_files_start_date=args.warc_files_start_date,
        warc_files_end_date=args.warc_files_end_date,
        strict_date=args.strict_date,
        reuse_previously_downloaded_files=args.
        reuse_previously_downloaded_files,
        local_download_dir_warc=os.path.join(args.outdir, 'warc'),
        continue_after_error=args.continue_after_error,
        show_download_progress=args.show_download_progress,
        number_of_extraction_processes=args.number_of_extraction_processes,
        log_level=logging.WARNING,
        delete_warc_after_extraction=args.delete_warc_after_extraction,
        continue_process=args.continue_process,
        fetch_images=False,
    )
  if args.shard:
    source_dir = os.path.join(args.outdir, 'source')
    mkdir(source_dir)
    _shard_news(
        txt_dir,
        source_dir,
        args.num_shards,
        args.number_of_sharding_processes,
    )
    print('Dask source prepared at {} !'.format(source_dir))


def console_script():
  main(attach_args().parse_args())
