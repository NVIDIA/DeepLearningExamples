# NVIDIA

import os
import spacy

#spacy.prefer_gpu()
spacy.require_gpu()

input_file = os.environ['WORKING_DIR'] + '/intermediate_files/wikipedia.txt'
output_file = os.environ['WORKING_DIR'] + '/final_test_file_single/wikipedia.segmented.txt'

nlp = spacy.load('en_core_web_sm')

doc_seperator = "\n"

file_mem = []

print("Reading file into memory.")
with open(input_file) as ifile:
  for line in ifile:
    if line != "\n":
      file_mem.append(line)

print("File read.")
print("Starting nlp.pipe")
docs = nlp.pipe(file_mem, batch_size=1000)

print("Starting to write output")
with open(output_file, "w") as ofile:
  for item in docs:
    for sent in item.sents:
      if sent.text != "\n":
        ofile.write(sent.text + "\n")
