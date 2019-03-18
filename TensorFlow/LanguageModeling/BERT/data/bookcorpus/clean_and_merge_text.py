# NVIDIA

import glob
import os

output_file = os.environ['WORKING_DIR'] + '/intermediate_files/bookcorpus.txt'
download_path = os.environ['WORKING_DIR'] + '/download/'

with open(output_file, "w") as ofile:
  for filename in glob.glob(download_path + '*.txt', recursive=True):
    with open(filename, mode='r', encoding="utf-8-sig") as file:
      for line in file:
        if line.strip() != "":
          ofile.write(line.strip() + " ")
    ofile.write("\n\n ")
