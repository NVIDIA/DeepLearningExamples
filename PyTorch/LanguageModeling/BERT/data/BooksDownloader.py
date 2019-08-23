# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
import subprocess

class BooksDownloader:
    def __init__(self, save_path):
        self.save_path = save_path
        pass


    def download(self):
        bookscorpus_download_command = 'python3 /workspace/bookcorpus/download_files.py --list /workspace/bookcorpus/url_list.jsonl --out'
        bookscorpus_download_command += ' ' + self.save_path + '/bookscorpus'
        bookscorpus_download_command += ' --trash-bad-count'
        bookscorpus_download_process = subprocess.run(bookscorpus_download_command, shell=True, check=True)
