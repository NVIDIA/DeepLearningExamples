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


import csv


o = csv.reader(open("data/biobert/chemprot-data_treeLSTM/dev.tsv", "r"), delimiter="\t")
nv = csv.reader(open("data/biobert/ChemProt_NV/dev.tsv", "r"), delimiter="\t")

count = {}
for l, i in enumerate(nv):
    if l == 0:
        continue
    if count.get(i[0].split(".")[0], None) is None:
        count[i[0].split(".")[0]] = 0
    count[i[0].split(".")[0]] += 1

count_1 = {}
for i in o:
    if count_1.get(i[0], None) is None:
        count_1[i[0]] = 0
    count_1[i[0]] += 1

for k in count.keys():
    if count[k] != count_1[k]:
        print(k, count[k], count_1[k])


# import os
# import csv
# import zipfile
# import argparse


# class ChemProtTextFormatting:
#     """A basic formatter to preprocess the chemprot dataset.
#     """

#     def __init__(self, input_folder, output_folder):

#         chemprot_folder = input_folder
#         with zipfile.ZipFile(os.path.join(chemprot_folder, "ChemProt_Corpus.zip"), "r") as zip:
#             zip.extractall(chemprot_folder)

#         chemprot_folder = os.path.join(input_folder, "ChemProt_Corpus")

#         with zipfile.ZipFile(os.path.join(chemprot_folder, "chemprot_development.zip")) as zip:
#             zip.extractall(chemprot_folder)

#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)

#         self.format(os.path.join(chemprot_folder, "chemprot_development"),
#                     "chemprot_development_entities.tsv", "chemprot_development_relations.tsv",
#                     "chemprot_development_abstracts.tsv", os.path.join(output_folder, "dev.tsv"))

#         with zipfile.ZipFile(os.path.join(chemprot_folder, "chemprot_test_gs.zip")) as zip:
#             zip.extractall(chemprot_folder)
#         self.format(os.path.join(chemprot_folder, "chemprot_test_gs"),
#                     "chemprot_test_entities_gs.tsv", "chemprot_test_relations_gs.tsv",
#                     "chemprot_test_abstracts_gs.tsv", os.path.join(output_folder, "test.tsv"))

#         with zipfile.ZipFile(os.path.join(chemprot_folder, "chemprot_training.zip")) as zip:
#             zip.extractall(chemprot_folder)
#         self.format(os.path.join(chemprot_folder, "chemprot_training"),
#                     "chemprot_training_entities.tsv", "chemprot_training_relations.tsv",
#                     "chemprot_training_abstracts.tsv", os.path.join(output_folder, "train.tsv"))



#     def format(self, chemprot_path, entity_filename, relations_filename, abstracts_filename, output_filename):
#         """
#         Constructs ChemProt dataset for Relation Extraction.

#         Args:
#           chemprot_path: Path to files
#           entity_filename: Contains labelled mention annotations of chemical compounds and genes/proteins.
#                             <PMID> <EntityNumber> <Type of Entity> <Start Character offset> <End Character Offset> <Text String>
#           relations_filename: Contains a subset of chemical-protein relations annotations for the Chemprot dataset
#                             <PMID> <CPR Group> <EntityNumber1> <EntityNumber2>
#           abstracts_filename: Contains plain text CHEMPROT PubMed Data
#                             <PMID> <Title of the Article> <Abstract of the Article>
#           output_filename: Path to output file that will contain preprocessed data
#                             <PMID.EntityNumber1.EntityNumber2> <Preprocessed Sentence> <CPR Group>
#         """

#         data = {}
#         train_entities = csv.reader(open(os.path.join(chemprot_path, entity_filename),
#                                          mode="r"), delimiter="\t")
#         for entity in train_entities:
#             id = entity[0]
#             if data.get(id, None) is None:
#                 data[id] = {"relations":[], "entities":{}}
#             data[id]["entities"][entity[1]] = (int(entity[3]), int(entity[4]), entity[2])

#         train_relations=csv.reader(open(os.path.join(chemprot_path, relations_filename),
#                                    mode="r"), delimiter="\t")
#         for relation in train_relations:
#             try:
#                 id = relation[0]
#                 data[id]["relations"].append((relation[1], relation[2], relation[4].split("Arg1:")[-1], relation[5].split("Arg2:")[-1]))
#             except:
#                 print("invalid id")
#                 raise ValueError

#         with open(output_filename, 'w') as ofile:
#             train_abstracts = csv.reader(open(os.path.join(chemprot_path, abstracts_filename),
#                                               mode="r"), delimiter="\t")
#             owriter = csv.writer(ofile, delimiter='\t', lineterminator=os.linesep)
#             owriter.writerow(["index", "sentence", "label"])

#             num_sentences = 0
#             rejected = 0
#             for abstract in train_abstracts:
#                 id = abstract[0]
#                 line = abstract[1] + abstract[2]

#                 for relation in data[id]["relations"]:
#                     tag1 = relation[2]
#                     tag2 = relation[3]
#                     start = 0
#                     for sentence in line.split("."):
#                         end = start + len(sentence)
#                         if data[id]["entities"][tag1][0] >= start and data[id]["entities"][tag2][0] >= start and \
#                                 data[id]["entities"][tag1][1] <= end and data[id]["entities"][tag2][1] <= end:
#                             for offset_start, offset_end, word in sorted([(data[id]["entities"][tag1][0], data[id]["entities"][tag1][1], data[id]["entities"][tag1][2]),
#                                                       (data[id]["entities"][tag2][0], data[id]["entities"][tag2][1], data[id]["entities"][tag2][2])],
#                                                      reverse=True):
#                                 sentence = sentence[:offset_start-start-1] + "@" + word + "$" + sentence[offset_end-start-1:]
#                             sentence = sentence.strip()
#                             owriter.writerow([id+"."+tag1+"."+tag2, sentence, relation[0] if relation[1] == "Y " else "false"])
#                             num_sentences += 1
#                             if id == "10064839":
#                                 print(tag1, tag2, start, end, offset_start, offset_end, "yes")
#                             break
#                         else:
#                             rejected += 1
#                             if id == "10064839":
#                                 print(tag1, tag2, start, end, data[id]["entities"][tag1][0], data[id]["entities"][tag1][1], data[id]["entities"][tag2][0], data[id]["entities"][tag2][1])
#                         start = end + 1
#             print("Succesfully written {} samples to {}".format(num_sentences, output_filename))
#             print("Rejected are", rejected)


# if __name__=="__main__":
#     parser = argparse.ArgumentParser(
#         description='Preprocessing Application for ChemProt'
#     )

#     parser.add_argument(
#         '--input_folder',
#         type=str,
#         help='Specify the input files in a comma-separated list (no spaces)'
#     )
#     parser.add_argument(
#         '--output_folder',
#         type=str,
#         help='Specify the input files in a comma-separated list (no spaces)'
#     )


#     args = parser.parse_args()
#     preprocess_chemprot = ChemProtTextFormatting(args.input_folder, args.output_folder)

#     # Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#     # Licensed under the Apache License, Version 2.0 (the "License");
#     # you may not use this file except in compliance with the License.
#     # You may obtain a copy of the License at
#     #
#     #     http://www.apache.org/licenses/LICENSE-2.0
#     #
#     # Unless required by applicable law or agreed to in writing, software
#     # distributed under the License is distributed on an "AS IS" BASIS,
#     # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     # See the License for the specific language governing permissions and
#     # limitations under the License.

#     import os
#     import csv
#     import zipfile
#     import argparse


#     class ChemProtTextFormatting:
#         """A basic formatter to preprocess the chemprot dataset.
#         """

#         def __init__(self, input_folder, output_folder):

#             chemprot_folder = input_folder
#             with zipfile.ZipFile(os.path.join(chemprot_folder, "ChemProt_Corpus.zip"), "r") as zip:
#                 zip.extractall(chemprot_folder)

#             chemprot_folder = os.path.join(input_folder, "ChemProt_Corpus")

#             with zipfile.ZipFile(os.path.join(chemprot_folder, "chemprot_development.zip")) as zip:
#                 zip.extractall(chemprot_folder)

#             if not os.path.exists(output_folder):
#                 os.makedirs(output_folder)

#             self.format(os.path.join(chemprot_folder, "chemprot_development"),
#                         "chemprot_development_entities.tsv", "chemprot_development_relations.tsv",
#                         "chemprot_development_abstracts.tsv", os.path.join(output_folder, "dev.tsv"))

#             with zipfile.ZipFile(os.path.join(chemprot_folder, "chemprot_test_gs.zip")) as zip:
#                 zip.extractall(chemprot_folder)
#             self.format(os.path.join(chemprot_folder, "chemprot_test_gs"),
#                         "chemprot_test_entities_gs.tsv", "chemprot_test_relations_gs.tsv",
#                         "chemprot_test_abstracts_gs.tsv", os.path.join(output_folder, "test.tsv"))

#             with zipfile.ZipFile(os.path.join(chemprot_folder, "chemprot_training.zip")) as zip:
#                 zip.extractall(chemprot_folder)
#             self.format(os.path.join(chemprot_folder, "chemprot_training"),
#                         "chemprot_training_entities.tsv", "chemprot_training_relations.tsv",
#                         "chemprot_training_abstracts.tsv", os.path.join(output_folder, "train.tsv"))

#         def format(self, chemprot_path, entity_filename, relations_filename, abstracts_filename, output_filename):
#             """
#             Constructs ChemProt dataset for Relation Extraction.

#             Args:
#               chemprot_path: Path to files
#               entity_filename: Contains labelled mention annotations of chemical compounds and genes/proteins.
#                                 <PMID> <EntityNumber> <Type of Entity> <Start Character offset> <End Character Offset> <Text String>
#               relations_filename: Contains a subset of chemical-protein relations annotations for the Chemprot dataset
#                                 <PMID> <CPR Group> <EntityNumber1> <EntityNumber2>
#               abstracts_filename: Contains plain text CHEMPROT PubMed Data
#                                 <PMID> <Title of the Article> <Abstract of the Article>
#               output_filename: Path to output file that will contain preprocessed data
#                                 <PMID.EntityNumber1.EntityNumber2> <Preprocessed Sentence> <CPR Group>
#             """

#             data = {}
#             train_entities = csv.reader(open(os.path.join(chemprot_path, entity_filename),
#                                              mode="r"), delimiter="\t")
#             for entity in train_entities:
#                 id = entity[0]
#                 if data.get(id, None) is None:
#                     data[id] = {"relations": {}, "entities": {"CHEMICAL": {"00": (0, 0, None)}, "GENE": {}}}
#                 data[id]["entities"]["CHEMICAL" if entity[2] == "CHEMICAL" else "GENE"][entity[1]] = (
#                 int(entity[3]), int(entity[4]), entity[2])

#             train_relations = csv.reader(open(os.path.join(chemprot_path, relations_filename),
#                                               mode="r"), delimiter="\t")
#             for relation in train_relations:
#                 try:
#                     id = relation[0]
#                     data[id]["relations"][(relation[4].split("Arg1:")[-1], relation[5].split("Arg2:")[-1])] = relation[
#                         1] if relation[2] == "Y " else "false"
#                 except:
#                     print("invalid id")
#                     raise ValueError
#             # print(data[list(data.keys())[0]])

#             with open(output_filename, 'w') as ofile:
#                 train_abstracts = csv.reader(open(os.path.join(chemprot_path, abstracts_filename),
#                                                   mode="r"), delimiter="\t")
#                 owriter = csv.writer(ofile, delimiter='\t', lineterminator=os.linesep)
#                 owriter.writerow(["index", "sentence", "label"])

#                 num_sentences = 0
#                 rejected = 0
#                 for abstract in train_abstracts:
#                     id = abstract[0]
#                     line = abstract[1] + abstract[2]

#                     for tag1 in data[id]["entities"]["CHEMICAL"].keys():
#                         for tag2 in data[id]["entities"]["GENE"].keys():
#                             relation = data[id]["relations"].get((tag2, tag1), None)
#                             relation = data[id]["relations"].get((tag1, tag2), None) if relation is None else relation
#                             if relation is None:
#                                 relation = "false"

#                             start = 0
#                             for sentence in line.split("."):
#                                 original_sentence = sentence
#                                 end = start + len(sentence)
#                                 tag1_details = data[id]["entities"]["CHEMICAL"][tag1]
#                                 tag2_details = data[id]["entities"]["GENE"][tag2]

#                                 if ((tag1_details[2] is None) or (
#                                         tag1_details[0] >= start and tag1_details[1] <= end)) and \
#                                         (tag2_details[0] >= start and tag2_details[1] <= end):
#                                     for offset_start, offset_end, value in sorted(
#                                             list(data[id]["entities"]["CHEMICAL"].values()) + list(
#                                                     data[id]["entities"]["GENE"].values()),
#                                             reverse=True):
#                                         if offset_start < start or offset_end > end or value is None:
#                                             continue
#                                         word = value if (offset_start, offset_end) == (
#                                         tag1_details[0], tag1_details[1]) or (offset_start, offset_end) == (
#                                                         tag2_details[0], tag2_details[1]) else "OTHER"
#                                         sentence = sentence[:offset_start - start - 1] + "@" + word + "$" + sentence[
#                                                                                                             offset_end - start - 1:]
#                                     sentence = sentence.strip()
#                                     owriter.writerow([id + "." + tag1 + "." + tag2, sentence, relation])
#                                     num_sentences += 1
#                                     # if id == list(data.keys())[0]:
#                                     #     print(original_sentence, sentence)
#                                     # break
#                                 else:
#                                     rejected += 1
#                                     if id == "10064839":
#                                         # print(tag1, tag2, start, end, tag1_details[0], tag1_details[1], tag2_details[0], tag2_details[1])
#                                         pass
#                                 start = end + 1
#                 print("Succesfully written {} samples to {}".format(num_sentences, output_filename))
#                 print("Rejected are", rejected)


#     if __name__ == "__main__":
#         parser = argparse.ArgumentParser(
#             description='Preprocessing Application for ChemProt'
#         )

#         parser.add_argument(
#             '--input_folder',
#             type=str,
#             help='Specify the input files in a comma-separated list (no spaces)'
#         )
#         parser.add_argument(
#             '--output_folder',
#             type=str,
#             help='Specify the input files in a comma-separated list (no spaces)'
#         )

#         args = parser.parse_args()
#         preprocess_chemprot = ChemProtTextFormatting(args.input_folder, args.output_folder)