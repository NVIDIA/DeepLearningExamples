PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '<\s>'

# special PAD, UNKNOWN, BEGIN-OF-STRING, END-OF-STRING tokens
PAD, UNK, BOS, EOS = [0, 1, 2, 3]

# path to the moses detokenizer, relative to the data directory
DETOKENIZER = 'mosesdecoder/scripts/tokenizer/detokenizer.perl'
