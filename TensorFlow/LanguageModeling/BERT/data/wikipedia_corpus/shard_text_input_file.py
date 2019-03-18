# NVIDIA

import os

input_file = os.environ['WORKING_DIR'] + '/final_text_file_single/wikipedia.segmented.nltk.txt'
output_file = os.environ['WORKING_DIR'] + '/final_text_files_sharded/wikipedia.segmented.part.'

doc_seperator = "\n"

line_buffer = []
shard_size = 396000 # Approximate, will split at next article break
line_counter = 0
shard_index = 0

ifile_lines = 0
with open(input_file) as ifile:
  for line in ifile:
    ifile_lines += 1

print("Input file contains", ifile_lines, "lines.")

iline_counter = 1
with open(input_file) as ifile:
  for line in ifile:
    if line_counter < shard_size and iline_counter < ifile_lines:
      line_buffer.append(line)
      line_counter += 1
      iline_counter += 1
    elif line_counter >= shard_size and line != "\n" and iline_counter < ifile_lines:
      line_buffer.append(line)
      line_counter += 1
      iline_counter += 1
    else:
       with open(output_file + str(shard_index) + ".txt", "w") as ofile:
         for oline in line_buffer:
           ofile.write(oline)
         line_buffer = []
         line_counter = 0
         shard_index += 1
