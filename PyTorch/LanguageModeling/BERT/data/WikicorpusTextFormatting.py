# NVIDIA

import glob
import os

class WikicorpusTextFormatting:
    def __init__(self, wiki_path, output_filename, recursive = False):
        self.wiki_path = wiki_path
        self.recursive = recursive
        self.output_filename = output_filename


    # This puts one article per line
    def merge(self):
        with open(self.output_filename, mode='w', newline='\n') as ofile:
            for dirname in glob.glob(self.wiki_path + '/*/', recursive=False):
                for filename in glob.glob(dirname + 'wiki_*', recursive=self.recursive):
                    print(filename)
                    article_lines = []
                    article_open = False

                    with open(filename, mode='r', newline='\n') as file:
                        for line in file:
                            if '<doc id=' in line:
                                article_open = True
                            elif '</doc>' in line:
                                article_open = False
                                for oline in article_lines[1:]:
                                    if oline != '\n':
                                        ofile.write(oline.rstrip() + " ")
                                ofile.write("\n\n")
                                article_lines = []
                            else:
                                if article_open:
                                    article_lines.append(line)