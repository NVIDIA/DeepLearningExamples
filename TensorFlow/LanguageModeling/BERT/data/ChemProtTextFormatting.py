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

import os
import csv
import zipfile
import argparse
import re

class ChemProtTextFormatting:
    """A basic formatter to preprocess the chemprot dataset.
    """

    def __init__(self, input_folder, output_folder):

        chemprot_folder = input_folder
        with zipfile.ZipFile(os.path.join(chemprot_folder, "ChemProt_Corpus.zip"), "r") as zip:
            zip.extractall(chemprot_folder)

        chemprot_folder = os.path.join(input_folder, "ChemProt_Corpus")

        with zipfile.ZipFile(os.path.join(chemprot_folder, "chemprot_development.zip")) as zip:
            zip.extractall(chemprot_folder)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.format(os.path.join(chemprot_folder, "chemprot_development"),
                    "chemprot_development_entities.tsv", "chemprot_development_relations.tsv",
                    "chemprot_development_abstracts.tsv", os.path.join(output_folder, "dev.tsv"))

        with zipfile.ZipFile(os.path.join(chemprot_folder, "chemprot_test_gs.zip")) as zip:
            zip.extractall(chemprot_folder)
        self.format(os.path.join(chemprot_folder, "chemprot_test_gs"),
                    "chemprot_test_entities_gs.tsv", "chemprot_test_relations_gs.tsv",
                    "chemprot_test_abstracts_gs.tsv", os.path.join(output_folder, "test.tsv"))

        with zipfile.ZipFile(os.path.join(chemprot_folder, "chemprot_training.zip")) as zip:
            zip.extractall(chemprot_folder)
        self.format(os.path.join(chemprot_folder, "chemprot_training"),
                    "chemprot_training_entities.tsv", "chemprot_training_relations.tsv",
                    "chemprot_training_abstracts.tsv", os.path.join(output_folder, "train.tsv"))



    def format(self, chemprot_path, entity_filename, relations_filename, abstracts_filename, output_filename):
        """
        Constructs ChemProt dataset for Relation Extraction.

        Args:
          chemprot_path: Path to files
          entity_filename: Contains labelled mention annotations of chemical compounds and genes/proteins.
                            <PMID> <EntityNumber> <Type of Entity> <Start Character offset> <End Character Offset> <Text String>
          relations_filename: Contains a subset of chemical-protein relations annotations for the Chemprot dataset
                            <PMID> <CPR Group> <EntityNumber1> <EntityNumber2>
          abstracts_filename: Contains plain text CHEMPROT PubMed Data
                            <PMID> <Title of the Article> <Abstract of the Article>
          output_filename: Path to output file that will contain preprocessed data
                            <PMID.EntityNumber1.EntityNumber2> <Preprocessed Sentence> <CPR Group>
        """

        data = {}
        train_entities = csv.reader(open(os.path.join(chemprot_path, entity_filename),
                                         mode="r"), delimiter="\t")
        for entity in train_entities:
            id = entity[0]
            if data.get(id, None) is None:
                data[id] = {"relations":{}, "entities":{"CHEMICAL":{}, "GENE":{}}}
            data[id]["entities"]["CHEMICAL" if entity[2] == "CHEMICAL" else "GENE"][entity[1]] = (int(entity[3]), int(entity[4]), entity[2])

        train_relations=csv.reader(open(os.path.join(chemprot_path, relations_filename),
                                   mode="r"), delimiter="\t")
        for relation in train_relations:
            try:
                id = relation[0]
                data[id]["relations"][(relation[4].split("Arg1:")[-1], relation[5].split("Arg2:")[-1])] = relation[1] if relation[2] == "Y " else "false"
            except:
                print("invalid id")
                raise ValueError
        # print(data[list(data.keys())[0]])

        with open(output_filename, 'w') as ofile:
            train_abstracts = csv.reader(open(os.path.join(chemprot_path, abstracts_filename),
                                              mode="r"), delimiter="\t")
            owriter = csv.writer(ofile, delimiter='\t', lineterminator=os.linesep)
            owriter.writerow(["index", "sentence", "label"])

            num_sentences = 0
            rejected = 0
            for abstract in train_abstracts:
                id = abstract[0]
                line = abstract[1] + "\n" + abstract[2]

                for tag1 in data[id]["entities"]["CHEMICAL"].keys():
                    for tag2 in data[id]["entities"]["GENE"].keys():
                        tag1_details = data[id]["entities"]["CHEMICAL"][tag1]
                        tag2_details = data[id]["entities"]["GENE"][tag2]
                        if ((tag1_details[0] <= tag2_details[0] and tag2_details[0] <= tag1_details[1]) # x1 <= y1 <= x2
                            or (tag1_details[0] <= tag2_details[1] and tag2_details[0] <= tag1_details[1])): # x1 <= y2 <= x2
                            continue

                        relation = data[id]["relations"].get((tag2, tag1), None)
                        relation = data[id]["relations"].get((tag1, tag2), None) if relation is None else relation
                        if relation is None:
                            relation = "false"

                        start = 0
                        line_protected = re.sub(r"(.)\.(?=[\d])", r"\1[PROTECTED_DOT]", line)
                        for sentence in re.split(r'\.|\?', line_protected):
                            sentence = sentence.replace("[PROTECTED_DOT]", ".")
                            original_sentence = sentence
                            end = start + len(sentence)

                            if (tag1_details[0] >= start and tag1_details[1] <= end) and \
                                    (tag2_details[0] >= start and tag2_details[1] <= end):
                                for offset_start, offset_end, value in sorted(list(data[id]["entities"]["CHEMICAL"].values()) + list(data[id]["entities"]["GENE"].values()),
                                                         reverse=True):
                                    if (offset_start, offset_end) == (tag1_details[0], tag1_details[1]) or (offset_start, offset_end) == (tag2_details[0], tag2_details[1]):
                                        if sentence[offset_start - start] == "@":
                                            offset_end = start + sentence.find('$',offset_start - start) + 1
                                        word = value
                                    elif offset_start < start or offset_end > end or sentence[offset_start - start] == "@":
                                        continue
                                    else:
                                        word = "OTHER"
                                    sentence = sentence[:offset_start-start] + "@" + word + "$" + sentence[offset_end-start:]
                                sentence = sentence.strip()
                                owriter.writerow([id+"."+tag1+"."+tag2, sentence, relation])
                                num_sentences += 1
                                if id == "23538201" and start == 1048:
                                    print("Accepted", tag1, tag2)

                            else:
                                rejected += 1

                            start = end + 1
            print("Succesfully written {} samples to {}".format(num_sentences, output_filename))
            print("Rejected are", rejected)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing Application for ChemProt'
    )

    parser.add_argument(
        '--input_folder',
        type=str,
        help='Specify the input files in a comma-separated list (no spaces)'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        help='Specify the input files in a comma-separated list (no spaces)'
    )


    args = parser.parse_args()
    preprocess_chemprot = ChemProtTextFormatting(args.input_folder, args.output_folder)