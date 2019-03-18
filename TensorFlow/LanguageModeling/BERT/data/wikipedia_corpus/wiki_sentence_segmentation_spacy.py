# NVIDIA

import os
import spacy

#spacy.prefer_gpu()
spacy.require_gpu()

input_file = os.environ['WORKING_DIR'] + '/intermediate_files/wikipedia.txt'
output_file = os.environ['WORKING_DIR'] + '/final_test_file_single/wikipedia.segmented.txt'

nlp = spacy.load('en_core_web_sm')

doc_seperator = "\n"

with open(input_file) as ifile:
  with open(output_file, "w") as ofile:
    for line in ifile:
      if line != "\n":
        doc = nlp(line)
        for sent in doc.sents:
          ofile.write(sent.text + "\n")
