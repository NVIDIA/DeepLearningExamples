# NVIDIA

import bz2
import os
import urllib.request
import sys

class WikiDownloader:
    def __init__(self, language, save_path):
        self.save_path = save_path + '/wikicorpus_' + language

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.language = language
        self.download_urls = {
            'en' : 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2',
            'zh' : 'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2'
        }

        self.output_files = {
            'en' : 'wikicorpus_en.xml.bz2',
            'zh' : 'wikicorpus_zh.xml.bz2'
        }


    def download(self):
        if self.language in self.download_urls:
            url = self.download_urls[self.language]
            file = self.output_files[self.language]

            print('Downloading:', url)
            if os.path.isfile(self.save_path + '/' + file):
                print('** Download file already exists, skipping download')
            else:
                response = urllib.request.urlopen(url)
                with open(self.save_path + '/' + file, "wb") as handle:
                    handle.write(response.read())

            # Always unzipping since this is relatively fast and will overwrite
            print('Unzipping:', self.output_files[self.language])
            #with open(self.save_path + '/' + file, mode='rb', buffering=131072) as f:
            #    it = iter(lambda: f.read(131072), b'')
            #    self.decompression(it, sys.stdout.buffer)

            zip = bz2.BZ2File(self.save_path + '/' + file)
            open(self.save_path + '/wikicorpus_' + self.language + '.xml', mode='wb', buffering=131072).write(zip.read())

        else:
            assert False, 'WikiDownloader not implemented for this language yet.'

    def decompression(self, input, output):
        decomp = bz2.BZ2Decompressor()

        for chunk in input:
            dc = decomp.decompress(chunk)
            output.write(dc)

