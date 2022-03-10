import os
import requests
import tqdm


def download(url, path, chunk_size=16 * 1024 * 1024):
  with requests.get(url, stream=True) as r:
    r.raise_for_status()
    total_size = int(r.headers.get('content-length', 0))
    progress_bar = tqdm.tqdm(total=total_size, unit='Bytes', unit_scale=True)
    with open(path, 'wb') as f:
      for chunk in r.iter_content(chunk_size=chunk_size):
        progress_bar.update(len(chunk))
        f.write(chunk)
    progress_bar.close()


def parse_str_of_num_bytes(s, return_str=False):
  try:
    power = 'kmg'.find(s[-1].lower()) + 1
    size = float(s[:-1]) * 1024**power
  except ValueError:
    raise ValueError('Invalid size: {}'.format(s))
  if return_str:
    return s
  else:
    return int(size)
