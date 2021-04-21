# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
import urllib.request
import tarfile

class GooglePretrainedWeightDownloader:
    def __init__(self, save_path):
        self.save_path = save_path + '/google_pretrained_weights'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Download urls
        self.model_urls = {
            'bert_base_uncased': ('https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12.tar.gz', 'uncased_L-12_H-768_A-12.tar.gz'),
            'bert_large_uncased': ('https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16.tar.gz', 'uncased_L-24_H-1024_A-16.tar.gz'),
            # 'bert_base_cased': ('https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/cased_L-12_H-768_A-12.tar.gz', 'cased_L-12_H-768_A-12.tar.gz'),
            # 'bert_large_cased': ('https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/cased_L-24_H-1024_A-16.tar.gz', 'cased_L-24_H-1024_A-16.tar.gz'),
            # 'bert_base_multilingual_cased': ('https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip', 'multi_cased_L-12_H-768_A-12.zip'),
            # 'bert_large_multilingual_uncased': ('https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip', 'multilingual_L-12_H-768_A-12.zip'),
            # 'bert_base_chinese': ('https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip', 'chinese_L-12_H-768_A-12.zip')
        }

        # SHA256sum verification for file download integrity (and checking for changes from the download source over time)
        self.bert_base_uncased_sha = {
            'bert_config.json': '7b4e5f53efbd058c67cda0aacfafb340113ea1b5797d9ce6ee411704ba21fcbc',
            'bert_model.ckpt.data-00000-of-00001': 'f8d2e9873133ea4d252662be01a074fb6b9e115d5fd1e3678d385cf65cf5210f',
            'bert_model.ckpt.index': '06a6b8cdff0e61f62f8f24946a607aa6f5ad9b969c1b85363541ab144f80c767',
            # 'checkpoint': 'da4c827756174a576abc3490e385fa8a36600cf5eb7bbea29315cf1f4ad59639',
            'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
        }

        self.bert_large_uncased_sha = {
            'bert_config.json': 'bfa42236d269e2aeb3a6d30412a33d15dbe8ea597e2b01dc9518c63cc6efafcb',
            'bert_model.ckpt.data-00000-of-00001': '9aa66efcbbbfd87fc173115c4f906a42a70d26ca4ca1e318358e4de81dbddb0b',
            'bert_model.ckpt.index': '1811d5b68b2fd1a8c5d2961b2691eb626d75c4e789079eb1ba3649aa3fff7336',
            # 'checkpoint': 'da4c827756174a576abc3490e385fa8a36600cf5eb7bbea29315cf1f4ad59639',
            'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
        }

        self.bert_base_cased_sha = {
            'bert_config.json': 'f11dfb757bea16339a33e1bf327b0aade6e57fd9c29dc6b84f7ddb20682f48bc',
            'bert_model.ckpt.data-00000-of-00001': 'ed0febc0fbcd2b7ef9f02112e00cb26c5de2086bca26c07b48b09c723446bc85',
            'bert_model.ckpt.index': 'af085a027ef3686466c9b662f9174129401bb4bc49856c917c02322ab7ca26d5',
            'checkpoint': 'da4c827756174a576abc3490e385fa8a36600cf5eb7bbea29315cf1f4ad59639',
            'vocab.txt': 'eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02',
        }

        self.bert_large_cased_sha = {
            'bert_config.json': '7adb2125c8225da495656c982fd1c5f64ba8f20ad020838571a3f8a954c2df57',
            'bert_model.ckpt.data-00000-of-00001': '1f96efeac7c8728e2bacb8ec6230f5ed42a26f5aa6b6b0a138778c190adf2a0b',
            'bert_model.ckpt.index': '373ed159af87775ce549239649bfc4df825bffab0da31620575dab44818443c3',
            'checkpoint': 'da4c827756174a576abc3490e385fa8a36600cf5eb7bbea29315cf1f4ad59639',
            'vocab.txt': 'eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02',
        }

        self.bert_base_multilingual_cased_sha = {
            'bert_config.json': 'e76c3964bc14a8bb37a5530cdc802699d2f4a6fddfab0611e153aa2528f234f0',
            'bert_model.ckpt.data-00000-of-00001': '55b8a2df41f69c60c5180e50a7c31b7cdf6238909390c4ddf05fbc0d37aa1ac5',
            'bert_model.ckpt.index': '7d8509c2a62b4e300feb55f8e5f1eef41638f4998dd4d887736f42d4f6a34b37',
            'bert_model.ckpt.meta': '95e5f1997e8831f1c31e5cf530f1a2e99f121e9cd20887f2dce6fe9e3343e3fa',
            'vocab.txt': 'fe0fda7c425b48c516fc8f160d594c8022a0808447475c1a7c6d6479763f310c',
        }

        self.bert_large_multilingual_uncased_sha = {
            'bert_config.json': '49063bb061390211d2fdd108cada1ed86faa5f90b80c8f6fdddf406afa4c4624',
            'bert_model.ckpt.data-00000-of-00001': '3cd83912ebeb0efe2abf35c9f1d5a515d8e80295e61c49b75c8853f756658429',
            'bert_model.ckpt.index': '87c372c1a3b1dc7effaaa9103c80a81b3cbab04c7933ced224eec3b8ad2cc8e7',
            'bert_model.ckpt.meta': '27f504f34f02acaa6b0f60d65195ec3e3f9505ac14601c6a32b421d0c8413a29',
            'vocab.txt': '87b44292b452f6c05afa49b2e488e7eedf79ea4f4c39db6f2f4b37764228ef3f',
        }

        self.bert_base_chinese_sha = {
            'bert_config.json': '7aaad0335058e2640bcb2c2e9a932b1cd9da200c46ea7b8957d54431f201c015',
            'bert_model.ckpt.data-00000-of-00001': '756699356b78ad0ef1ca9ba6528297bcb3dd1aef5feadd31f4775d7c7fc989ba',
            'bert_model.ckpt.index': '46315546e05ce62327b3e2cd1bed22836adcb2ff29735ec87721396edb21b82e',
            'bert_model.ckpt.meta': 'c0f8d51e1ab986604bc2b25d6ec0af7fd21ff94cf67081996ec3f3bf5d823047',
            'vocab.txt': '45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c',
        }

        # Relate SHA to urls for loop below
        self.model_sha = {
            'bert_base_uncased': self.bert_base_uncased_sha,
            'bert_large_uncased': self.bert_large_uncased_sha,
            # 'bert_base_cased': self.bert_base_cased_sha,
            # 'bert_large_cased': self.bert_large_cased_sha,
            # 'bert_base_multilingual_cased': self.bert_base_multilingual_cased_sha,
            # 'bert_large_multilingual_uncased': self.bert_large_multilingual_uncased_sha,
            # 'bert_base_chinese': self.bert_base_chinese_sha
        }

    # Helper to get sha256sum of a file
    def sha256sum(self, filename):
      h  = hashlib.sha256()
      b  = bytearray(128*1024)
      mv = memoryview(b)
      with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
          h.update(mv[:n])

      return h.hexdigest()

    def download(self):
        # Iterate over urls: download, unzip, verify sha256sum
        found_mismatch_sha = False
        for model in self.model_urls:
          url = self.model_urls[model][0]
          file = self.save_path + '/' + self.model_urls[model][1]

          print('Downloading', url)
          response = urllib.request.urlopen(url)
          with open(file, 'wb') as handle:
            handle.write(response.read())

          print('Unzipping', file)
          tf = tarfile.open(file)
          tf.extractall(self.save_path)

          sha_dict = self.model_sha[model]
          for extracted_file in sha_dict:
            sha = sha_dict[extracted_file]
            if sha != self.sha256sum(file[:-7] + '/' + extracted_file):
              found_mismatch_sha = True
              print('SHA256sum does not match on file:', extracted_file, 'from download url:', url)
            else:
              print(file[:-7] + '/' + extracted_file, '\t', 'verified')

        if not found_mismatch_sha:
          print("All downloads pass sha256sum verification.")

    def serialize(self):
        pass

    def deserialize(self):
        pass

    def listAvailableWeights(self):
        print("Available Weight Datasets")
        for item in self.model_urls:
            print(item)

    def listLocallyStoredWeights(self):
        pass

