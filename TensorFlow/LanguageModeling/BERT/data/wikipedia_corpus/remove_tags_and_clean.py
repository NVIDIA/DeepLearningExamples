# NVIDIA

import glob
import os

output_file = os.environ['WORKING_DIR'] + '/intermediate_files/wikipedia.txt'

with open(output_file, "w") as ofile:
  for dirname in glob.glob('extracted_articles/*/', recursive=False):
    for filename in glob.glob(dirname + 'wiki_*', recursive=True):
      print(filename)
      article_lines = []
      article_open = False
      
      with open(filename, "r") as file:
        for line in file:
          if "<doc id=" in line:
            article_open = True
          elif "</doc>" in line:
            article_open = False
            for oline in article_lines[1:]:
              if oline != "\n":
                ofile.write(oline.rstrip() + " ")
            ofile.write("\n\n")
            article_lines = []
          else:
            if article_open:
              article_lines.append(line)
            

