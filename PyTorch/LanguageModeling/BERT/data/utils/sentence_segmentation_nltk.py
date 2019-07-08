# NVIDIA

import argparse
import nltk
import os

nltk.download('punkt')

parser = argparse.ArgumentParser(description='Sentence Segmentation')

parser.add_argument('input_file', type=str)
parser.add_argument('output_file', type=str)

args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

doc_seperator = "\n"

with open(input_file) as ifile:
  with open(output_file, "w") as ofile:
    for line in ifile:
      if line != "\n":
        sent_list = nltk.tokenize.sent_tokenize(line)
        for sent in sent_list:
          ofile.write(sent + "\n")
        ofile.write(doc_seperator)
