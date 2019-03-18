# NVIDIA

import glob
import os
import random
import shutil

input_dir = os.environ['WORKING_DIR'] + '/final_text_files_sharded/'
output_dir = os.environ['WORKING_DIR'] + '/test_set_text_files/'

random.seed(13254)
n_shards_to_keep = 3

file_glob = glob.glob(input_dir + '/*', recursive=False)
file_glob = random.sample(file_glob, n_shards_to_keep)

for filename in file_glob:
  shutil.copy(filename, output_dir) 
