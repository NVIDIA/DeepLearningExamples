Distillation
========

To get setup to run Knowledge Distillation on BERT once in the container, run the following:

```
cd /workspace/bert/distillation
bash utils/perform_distillation_prereqs.sh
```

`perform_distillation_prereqs.sh` performs the following:
- Downloads and processes prerequisite BERT-base checkpoints to `/workspace/bert/distillation/checkpoints`
- Downloads prerequisite GloVe embeddings to `/workspace/bert/data/downloads/glove`

After performing prerequisite tasks, in the container run the following to produce fully distilled BERT models for SQuADv1.1 and SST-2.
```
bash run_e2e_distillation.sh
```

`run_e2e_distillation.sh` contains 8 command lines to obtain fully distilled BERT models for SQuADv1.1 and SST-2. The distilled BERT model has a config (N=4, D=312, Di=1200 , H=12). To distill knowledge into models of different sizes, a new `BERT_4L_312D/config.json` can be created and passed as a starting point in `run_e2e_distillation.sh`

`run_e2e_distillation.sh` contains the following:
- Phase1 distillation: Generic distillation on Wikipedia dataset of maximum sequence length 128. `--input_dir` needs to be update respectively.
- Phase2 distillation: Generic distillation on Wikipedia dataset of maximum sequence length 512. `--input_dir` needs to be update respectively.

*Task specific distillation: SQuAD v1.1* (maximum sequence length 384)
- Data augmentation
- Distillation on task specific SQuad v1.1 dataset using losses based on transformer backbone only
- Distillation on task specific SQuad v1.1 dataset using loss based on task specific prediction head only.

*Task specific distillation: SST-2* (maximum sequence length 128)
- Data augmentation
- Distillation on task specific SST-2 dataset using losses based on transformer backbone only
- Distillation on task specific SST-2 dataset using loss based on task specific prediction head only.

![BERT Distillation Flow](https://developer.nvidia.com/sites/default/files/akamai/joc_model.png)

Note: Task specific distillation for SST-2 uses as output checkpoint of phase1 distillation as starting point, whereas task specific distillation of SQuAD v1.1 uses output checkpoint of phase2 distillation as a starting point.

One can download different general and task-specific distilled checkpoints from NGC:
| Model                  | Description                                                               |
|------------------------|---------------------------------------------------------------------------|
| [bert-dist-4L-288D-uncased-qa](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_4l_288d_qa_squad11_amp/files) | 4 layer distilled model fine-tuned on SQuAD v1.1                                         |
| [bert-dist-4L-288D-uncased-sst2](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_4l_288d_ft_sst2_amp/files) | 4 layer distilled model fine-tuned on GLUE SST-2                                       |
| [bert-dist-4L-288D-uncased-pretrained](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_4l_288d_pretraining_amp/files) | 4 layer distilled model pretrained checkpoint on Generic corpora like Wikipedia. |
| [bert-dist-6L-768D-uncased-qa](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_6l_768d_qa_squad11_amp/files) | 6 layer distilled model fine-tuned on SQuAD v1.1                                         |
| [bert-dist-6L-768D-uncased-sst2](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_6l_768d_ft_sst2_amp/files) | 6 layer distilled model fine-tuned on GLUE SST-2                                       |
| [bert-dist-6L-768D-uncased-pretrained](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_6l_768d_pretraining_amp/files) | 6 layer distilled model pretrained checkpoint on Generic corpora like Wikipedia. |


Following results were obtained on NVIDIA DGX-1 with 32G on pytorch:20.12-py3 NGC container.

*Accuracy achieved and E2E time to train on NVIDIA DGX-1 With 32G:*

| Student         | Task             | SubTask          | Time(hrs)  | Total Time (hrs)| Accuracy | BERT Base Accuracy  |
| --------------- |:----------------:| :---------------:| :--------: | :-------------: | :------: | ------------------: |
| 4 Layers; H=288 | Distil Phase 1   | backbone loss    | 1.399      |                 |          |                     |
| 4 Layers; H=288 | Distil Phase 2   | backbone loss    | 0.649      |                 |          |                     |
| 4 Layers; H=288 | Distil SST-2     | backbone loss    | 1.615      |                 |          |                     |
| 4 Layers; H=288 | Distil SST-2     | final layer loss | 0.469      | 3.483           | 90.82    | 91.51               |
| 4 Layers; H=288 | Distil SQuADv1.1 | backbone loss    | 3.471      |                 |          |                     |
| 4 Layers; H=288 | Distil SQuADv1.1 | final layer loss | 3.723      | 9.242           | 83.09    | 88.58                |
| 6 Layers; H=768 | SST-2            |                  |            |                 | 91.97    | 91.51               |
| 6 Layers; H=768 | SQuADv1.1        |                  |            |                 | 88.43    | 88.58                |

To perform inference refer to [Inference Performance Benchmark](../#inference-process)

*FP16 Inference Performance:*

| Model                  | BS     | Infer Perf (seqlen128) (seq/sec)| Infer Perf (seqlen384) (seq/sec) | Speedup vs Bert Large (seqlen128)| Speedup vs Bert Large (seqlen384)| Speedup vs Bert Base (seqlen128) | Speedup vs Bert Base (seqlen384) |
| ---------------------  |:------:| :----------------------------:  | :----------------------------:   | :------------------------------: | :------------------------------: | :------------------------------: | -------------------------------- |
| BERT Large PyT         |8       | 502                             | 143                              | 1                                | 1                                | 0.3625                           | 0.333                            |
| BERT Base PyT          |128     | 1385                            | 429                              | 2.7590                           | 3                                | 1                                | 1                                |
| NV_DistillBERT_4l_312D |128     | 13600                           | 2300                             | 27.0916                          | 16.0839                          | 9.8195                           | 5.36130                          |


