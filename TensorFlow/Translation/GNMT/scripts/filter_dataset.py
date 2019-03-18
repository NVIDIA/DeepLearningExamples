# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
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
import argparse
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description='Clean dataset')
    parser.add_argument('-f1', '--file1', help='file1')
    parser.add_argument('-f2', '--file2', help='file2')
    return parser.parse_args()


def save_output(fname, data):
    with open(fname, 'w') as f:
        f.writelines(data)


def main():
    """
    Discards all pairs of sentences which can't be decoded by latin-1 encoder.

    It aims to filter out sentences with rare unicode glyphs and pairs which
    are most likely not valid English-German sentences.

    Examples of discarded sentences:

        ✿★★★Hommage au king de la pop ★★★✿ ✿★★★Que son âme repos...

        Для их осуществления нам, прежде всего, необходимо преодолеть
        возражения рыночных фундаменталистов, которые хотят ликвидировать или
        уменьшить роль МВФ.

        practised as a scientist in various medical departments of the ⇗Medical
        University of Hanover , the ⇗University of Ulm , and the ⇗RWTH Aachen
        (rheumatology, pharmacology, physiology, pathology, microbiology,
        immunology and electron-microscopy).

        The same shift】 and press 【】 【alt out with a smaller diameter
        circle.

        Brought to you by ABMSUBS ♥leira(Coordinator/Translator)
        ♥chibichan93(Timer/Typesetter) ♥ja...

        Some examples: &0u - ☺ &0U - ☻ &tel - ☏ &PI - ¶ &SU - ☼ &cH- - ♥ &M2=♫
        &sn - ﺵ SGML maps SGML to unicode.
    """
    args = parse_args()

    c = Counter()
    skipped = 0
    valid = 0
    data1 = []
    data2 = []

    with open(args.file1) as f1, open(args.file2) as f2:
        for idx, lines in enumerate(zip(f1, f2)):
            line1, line2 = lines
            if idx % 100000 == 1:
                print('Processed {} lines'.format(idx))
            try:
                line1.encode('latin1')
                line2.encode('latin1')
            except UnicodeEncodeError:
                skipped += 1
            else:
                data1.append(line1)
                data2.append(line2)
                valid += 1
                c.update(line1)

    ratio = valid / (skipped + valid)
    print('Skipped: {}, Valid: {}, Valid ratio {}'.format(skipped, valid, ratio))
    print('Character frequency:', c)

    save_output(args.file1, data1)
    save_output(args.file2, data2)


if __name__ == '__main__':
    main()
