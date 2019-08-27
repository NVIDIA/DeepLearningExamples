# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.

import bz2
import os
import urllib.request
import sys

class MRPCDownloader:
    def __init__(self, save_path):
        self.save_path = save_path + '/mrpc'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Documentation - Download link obtained from here: https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py
        self.download_urls = {
            'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc' : 'mrpc_dev_ids.tsv'
        }

    def download(self):
        for item in self.download_urls:
            url = item
            file = self.download_urls[item]

            print('Downloading:', url)
            if os.path.isfile(self.save_path + '/' + file):
                print('** Download file already exists, skipping download')
            else:
                response = urllib.request.urlopen(url)
                with open(self.save_path + '/' + file, "wb") as handle:
                    handle.write(response.read())


