# NVIDIA

import nltk
import os

nltk.download('punkt')

input_file = os.environ['WORKING_DIR'] + '/intermediate_files/wikipedia.txt'
output_file = os.environ['WORKING_DIR'] + '/final_text_file_single/wikipedia.segmented.nltk.txt'

doc_seperator = "\n"

with open(input_file) as ifile:
  with open(output_file, "w") as ofile:
    for line in ifile:
      if line != "\n":
        sent_list = nltk.tokenize.sent_tokenize(line)
        for sent in sent_list:
          ofile.write(sent + "\n")
        ofile.write(doc_seperator)
