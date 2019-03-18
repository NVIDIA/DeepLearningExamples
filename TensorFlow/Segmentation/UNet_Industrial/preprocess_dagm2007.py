#!/usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
# Copyright (c) Jonathan Dekhtiar - contact@jonathandekhtiar.eu
# All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

import os
import glob
import ntpath

import argparse

from collections import defaultdict

parser = argparse.ArgumentParser(description="DAGM2007_preprocessing")

parser.add_argument('--data_dir', required=True, type=str, help="Path to DAGM 2007 private dataset")

DEFECTIVE_COUNT = defaultdict(lambda: defaultdict(int))

EXPECTED_DEFECTIVE_SAMPLES_PER_CLASS = {
    "Train": {
        1: 79,
        2: 66,
        3: 66,
        4: 82,
        5: 70,
        6: 83,
        7: 150,
        8: 150,
        9: 150,
        10: 150,
    },
    "Test": {
        1: 71,
        2: 84,
        3: 84,
        4: 68,
        5: 80,
        6: 67,
        7: 150,
        8: 150,
        9: 150,
        10: 150,
    }
}

if __name__ == "__main__":

    FLAGS, unknown_args = parser.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    if not os.path.exists(FLAGS.data_dir):
        raise ValueError('The dataset directory received `%s` does not exists' % FLAGS.data_dir)

    for challenge_id in range(10):
        challenge_name = "Class%d" % (challenge_id + 1)
        challenge_folder_path = os.path.join(FLAGS.data_dir, challenge_name)

        print("[DAGM Preprocessing] Parsing Class ID: %02d ..." % (challenge_id + 1))

        if not os.path.exists(challenge_folder_path):
            raise ValueError('The folder `%s` does not exists' % challenge_folder_path)

        for data_set in ["Train", "Test"]:

            challenge_set_folder_path = os.path.join(challenge_folder_path, data_set)

            if not os.path.exists(challenge_set_folder_path):
                raise ValueError('The folder `%s` does not exists' % challenge_set_folder_path)

            with open(os.path.join(challenge_folder_path, "%s_list.csv" % data_set.lower()), 'w') as data_list_file:
                data_list_file.write('image_filepath,lbl_image_filepath,is_defective\n')

                files = glob.glob(os.path.join(challenge_set_folder_path, "*.PNG"))

                for file in files:
                    filepath, fullname = ntpath.split(file)

                    filename, extension = os.path.splitext(os.path.basename(fullname))

                    lbl_filename = "%s_label.PNG" % filename
                    lbl_filepath = os.path.join(filepath, "Label", lbl_filename)

                    if os.path.exists(lbl_filepath):
                        defective = True
                    else:
                        defective = False
                        lbl_filename = ""

                    if defective:
                        DEFECTIVE_COUNT[data_set][challenge_id + 1] += 1

                    data_list_file.write('%s,%s,%d\n' % (fullname, lbl_filename, defective))

                if DEFECTIVE_COUNT[data_set][challenge_id +
                                             1] != EXPECTED_DEFECTIVE_SAMPLES_PER_CLASS[data_set][challenge_id + 1]:
                    raise RuntimeError(
                        "There should be `%d` defective samples instead of `%d` in challenge (%s): %d" % (
                            DEFECTIVE_COUNT[data_set][challenge_id + 1],
                            EXPECTED_DEFECTIVE_SAMPLES_PER_CLASS[data_set][challenge_id + 1], data_set, challenge_id + 1
                        )
                    )
