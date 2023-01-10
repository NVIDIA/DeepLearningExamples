# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path


# Define val and test; the remaining ones will be train IDs
val_ids = {
    'com_SF_ce227', 'com_SF_ce832', 'com_SF_ce912','com_SF_ce979',
    'com_SF_ce998', 'com_SF_ce1045', 'com_SF_ce1282','com_SF_ce1329',
    'com_SF_ce1350', 'com_SF_ce1376', 'com_SF_ce1519','com_SF_ce1664',
    'com_SF_ce1777', 'com_SF_ce1843', 'com_SF_ce2017','com_SF_ce2042',
    'com_SF_ce2100', 'com_SF_ce2251', 'com_SF_ce2443','com_SF_ce2566',
}

test_ids = {
    'com_SF_ce161', 'com_SF_ce577', 'com_SF_ce781', 'com_SF_ce814',
    'com_SF_ce1042', 'com_SF_ce1089', 'com_SF_ce1123', 'com_SF_ce1425',
    'com_SF_ce1514', 'com_SF_ce1577', 'com_SF_ce1780', 'com_SF_ce1857',
    'com_SF_ce1940', 'com_SF_ce2051', 'com_SF_ce2181', 'com_SF_ce2258',
    'com_SF_ce2406', 'com_SF_ce2512', 'com_SF_ce2564', 'com_SF_ce2657'
}


def generate(fpath, ids_text, pitch=True, text=True):

    with open(fpath, 'w') as f:
        for id_, txt in ids_text.items():
            row = f"wavs/{id_}.wav"
            row += "|" + f"pitch/{id_}.pt" if pitch else ""
            row += "|" + txt if text else ""
            f.write(row + "\n")


def generate_inference_tsv(fpath, ids_text):

    with open(fpath, 'w') as f:
        f.write("output\ttext\n")
        for id_, txt in ids_text.items():
            f.write(f"{id_}.wav\t{txt}\n")


def main():
    parser = argparse.ArgumentParser(
        description='SF bilingual dataset filelists generator')
    parser.add_argument('transcripts', type=Path, default='./text_SF.txt',
                        help='Path to LJSpeech dataset metadata')
    parser.add_argument('output_dir', default='data/filelists', type=Path,
                        help='Directory to generate filelists to')
    args = parser.parse_args()

    with open(args.transcripts) as f:
        # A dict of ID:transcript pairs
        transcripts = dict(line.replace("\ufeff", "").replace("－", "-").strip().split(' ', 1)
                           for line in f)
    transcripts = {id_.replace("com_DL", "com_SF"): text.lower()
                   for id_, text in transcripts.items()}

    val_ids_text = {id_: transcripts[id_] for id_ in val_ids}
    test_ids_text = {id_: transcripts[id_] for id_ in test_ids}
    train_ids_text = {id_: transcripts[id_] for id_ in transcripts
                      if id_ not in test_ids and id_ not in val_ids}

    prefix = Path(args.output_dir, "sf_audio_pitch_text_")
    generate(str(prefix) + "val.txt", val_ids_text)
    generate(str(prefix) + "test.txt", test_ids_text)
    generate(str(prefix) + "train.txt", train_ids_text)

    prefix = Path(args.output_dir, "sf_audio_")
    generate(str(prefix) + "val.txt", val_ids_text, False, False)
    generate(str(prefix) + "test.txt", test_ids_text, False, False)
    generate(str(prefix) + "train.txt", train_ids_text, False, False)

    # train + val + test for pre-processing
    generate(Path(args.output_dir, "sf_audio_text.txt"),
             {**val_ids_text, **test_ids_text, **train_ids_text}, False, True)

    generate_inference_tsv(Path(args.output_dir, "sf_test.tsv"), test_ids_text)


if __name__ == '__main__':
    main()
