# NVIDIA

import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Cleaning and merge downloaded bookcorpus files')

parser.add_argument('download_path', type=str)
parser.add_argument('output_file', type=str)

args = parser.parse_args()

download_path = args.download_path
output_file = args.output_file

with open(output_file, "w") as ofile:
  for filename in glob.glob('{}/*.txt'.format(download_path), recursive=True):
    with open(filename, mode='r', encoding="utf-8-sig") as file:
      for line in file:
        if line.strip() != "":
          ofile.write(line.strip() + " ")
    ofile.write("\n\n")
