## BERT for PyTorch on LaunchPad

The instructions to run BERT on LaunchPad Base Command (BCP) can be found here:

<https://docs.nvidia.com/launchpad/ai/prod_base-command-nlp.html>

Refer to the main [README](README.md) instructions and LaunchPad instructions.

On BCP create a workspace where the data, pretrain shards, logs, checkpoints,
and models will be stored.
```
ngc workspace create --name bert_example <other args for ACE ORG TEAM ...>
```

#### Download the dataset

Run the following ngc job to download data to the workspace.
```
ngc batch run \
  --name "create_bert_dataset" \
  --ace <ACE> \
  --org <ORG> \
  --team <TEAM> \
  --instance dgxa100.80g.1.norm \
  --image <BERT CONTAINER URI> \
  --result /results \
  --workspace bert_example:/bert_example:RW \
  --total-runtime 3h \
  --commandline "\
set -x && \
export BERT_PREP_WORKING_DIR=/bert_example/bertdata
mkdir -p \${BERT_PREP_WORKING_DIR}
bash -x /workspace/bert/data/create_datasets_from_start.sh
"
```


#### Multi-node

Equivalent to `run.sub` script, the corresponding script to run on LaunchPad BCP
is provided `run_bcp.sub`. Let's assume that a workspace "bert_example" has been
created. Run the following commands for training.

Phase 1 training.
```
ngc batch run \
  --name "run_bert_phase1_2node" \
  --ace <ACE> \
  --org <ORG> \
  --team <TEAM or no-team> \
  --instance dgxa100.80g.8.norm \
  --array-type PYTORCH \
  --replicas 4 \
  --image <BERT CONTAINER URI> \
  --result /results \
  --workspace bert_example:/bert_example:RW \
  --total-runtime 20h \
  --commandline "\
set -x && \
export CODEDIR=/workspace/bert && \
export DATADIR=/bert_example/bertdata && \
export PHASE=1 && \
export OUTPUTDIR=/bert_example/bertresults/phase\${PHASE}_jobid_\${NGC_JOB_ID} && \
export PRETRAINDIR=/bert_example/bertpretrain && \
mkdir -p \$OUTPUTDIR && \
bash -x \${CODEDIR}/run_bcp.sh
"
```

Phase 2 training.
```
ngc batch run \
  --name "run_bert_phase2_2node" \
  --ace <ACE> \
  --org <ORG> \
  --team <TEAM or no-team> \
  --instance dgxa100.80g.8.norm \
  --array-type PYTORCH \
  --replicas 4 \
  --image <BERT CONTAINER URI> \
  --result /results \
  --workspace bert_example:/bert_example:RW \
  --total-runtime 20h \
  --commandline "\
set -x && \
export CODEDIR=/workspace/bert && \
export DATADIR=/bert_example/bertdata && \
export PHASE=2 && \
export PHASE1JOBID=<PHASE1 JOB ID> && \
export OUTPUTDIR=/bert_example/bertresults/phase\${PHASE}_jobid_\${NGC_JOB_ID} && \
export PRETRAINDIR=/bert_example/bertpretrain && \
export INIT_CHECKPOINT=/bert_example/bertresults/phase1_jobid_\${PHASE1JOBID}/checkpoints/ckpt_7038.pt && \
mkdir -p \$OUTPUTDIR && \
bash -x \${CODEDIR}/run_bcp.sh
"
```

The example commands above use 4 nodes of DGX A100s. Refer to the `run_bcp.sub`
script for details on other parameters.
