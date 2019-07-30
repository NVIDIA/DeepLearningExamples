# NVIDIA

import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Cleaning and merge downloaded bookcorpus files')

parser.add_argument('extracted_articles_path', type=str)
parser.add_argument('output_file', type=str)

args = parser.parse_args()

extracted_articles_path = args.extracted_articles_path
output_file = args.output_file

with open(output_file, "w") as ofile:
  for dirname in glob.glob('{}/*/'.format(extracted_articles_path), recursive=False):
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
            

