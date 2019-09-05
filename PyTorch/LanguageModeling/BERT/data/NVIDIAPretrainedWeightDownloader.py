# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.

import os

class NVIDIAPretrainedWeightDownloader:
    def __init__(self, save_path):
        self.save_path = save_path + '/nvidia_pretrained_weights'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        pass


    def download(self):
        assert False, 'NVIDIAPretrainedWeightDownloader not implemented yet.'