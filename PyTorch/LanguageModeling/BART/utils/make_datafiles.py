# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import os
import hashlib


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

finished_files_dir = "cnn_dm"

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode())
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string
  abstract = ' '.join(highlights)

  return article, abstract


def write_to_bin(url_file, out_name):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print("Making bin file for URLs listed in %s..." % url_file)
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  story_fnames = [s+".story" for s in url_hashes]
  num_stories = len(story_fnames)

  article_out = out_name + '.source'
  abstract_out = out_name + '.target'

  with open(article_out, 'w') as article_writer, open(abstract_out, 'w') as abstract_writer:
    for idx,s in enumerate(story_fnames):
      if idx % 1000 == 0:
        print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(cnn_stories_dir, s)):
        story_file = os.path.join(cnn_stories_dir, s)
      elif os.path.isfile(os.path.join(dm_stories_dir, s)):
        story_file = os.path.join(dm_stories_dir, s)
      else:
        print("Error: Couldn't find story file %s in story directories %s and %s." % (s, cnn_stories_dir, dm_stories_dir))
        # Check again if stories directories contain correct number of files
        print("Checking that the stories directories %s and %s contain correct number of files..." % (cnn_stories_dir, dm_stories_dir))
        check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
        check_num_stories(dm_stories_dir, num_expected_dm_stories)
        raise Exception("Stories directories %s and %s contain correct number of files but story file %s found in neither." % (cnn_stories_dir, dm_stories_dir, s))

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(story_file)

      article_writer.write(article + '\n')
      abstract_writer.write(abstract + '\n')

  print("Finished writing file %s and %s\n" % (article_out, abstract_out))

def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("USAGE: python make_datafiles.py <cnn_stories_dir> <dailymail_stories_dir>")
    sys.exit()
  cnn_stories_dir = sys.argv[1]
  dm_stories_dir = sys.argv[2]

  # Check the stories directories contain the correct number of .story files
  check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
  check_num_stories(dm_stories_dir, num_expected_dm_stories)

  # Create some new directories
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test"))
  write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val"))
  write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train"))

