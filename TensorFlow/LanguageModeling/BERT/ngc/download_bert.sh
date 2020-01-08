#!/bin/bash
# DeepLearningExamples/TensorFlow/LanguageModeling/BERT
BERT_DIR=".."
CONFIG_DIR="${BERT_DIR}/config.qa"
mkdir -p ${BERT_DIR}/data/finetuned_model_fp16
wget -nc -q --show-progress -O ${BERT_DIR}/data/finetuned_model_fp16/bert_tf_v2_large_fp16_384.zip \
        https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_v2_large_fp16_384/versions/1/zip
unzip -n -d ${BERT_DIR}/data/finetuned_model_fp16/ ${BERT_DIR}/data/finetuned_model_fp16/bert_tf_v2_large_fp16_384.zip

wget -nc --show-progress -O bert_scripts.zip \
         https://api.ngc.nvidia.com/v2/recipes/nvidia/bert_for_tensorflow/versions/1/zip
mkdir -p ${BERT_DIR}
unzip -n -d ${BERT_DIR} bert_scripts.zip

mkdir -p ${CONFIG_DIR}
wget -nc https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt \
        -O ${CONFIG_DIR}/vocab.txt

cat >> ${CONFIG_DIR}/bert_config.json << EOF
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "max_position_embeddings": 512,
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
EOF

cat >> ${CONFIG_DIR}/input.json << EOF
{"data": 
 [
     {"title": "Project Apollo",
      "paragraphs": [
          {"context":"The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's national goal of landing a man on the Moon and returning him safely to the Earth by the end of the 1960s, which he proposed in a May 25, 1961, address to Congress. Project Mercury was followed by the two-man Project Gemini. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to 1972, and was supported by the two man Gemini program which ran concurrently with it from 1962 to 1966. Gemini missions developed some of the space travel techniques that were necessary for the success of the Apollo missions. Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications Program, which consisted of Skylab, a space station that supported three manned missions in 1973-74, and the Apollo-Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975.", 
           "qas": [
               { "question": "What project put the first Americans into space?", 
                 "id": "Q1"
               },
               { "question": "What program was created to carry out these projects and missions?",
                 "id": "Q2"
               },
               { "question": "What year did the first manned Apollo flight occur?",
                 "id": "Q3"
               },                
               { "question": "What President is credited with the notion of putting Americans on the moon?",
                 "id": "Q4"
               },
               { "question": "Who did the U.S. collaborate with on an Earth orbit mission in 1975?",
                 "id": "Q5"
               },
               { "question": "How long did Project Apollo run?",
                 "id": "Q6"
               },               
               { "question": "What program helped develop space travel techniques that Project Apollo used?",
                 "id": "Q7"
               },                
               {"question": "What space station supported three manned missions in 1973-1974?",
                 "id": "Q8"
               }
]}]}]}
EOF
