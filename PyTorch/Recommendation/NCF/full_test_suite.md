rm -r /data/cache/ml-20m

## Prepare the standard dataset:

./prepare_dataset.sh

## Prepare the modified dataset:

./test_dataset.sh

## Run on the modified dataset:

./test_cases.sh

## Check featurespec:

python test_featurespec_correctness.py /data/cache/ml-20m/feature_spec.yaml /data/ml-20m/feature_spec_template.yaml

## Other dataset:

rm -r /data/cache/ml-1m

./prepare_dataset.sh ml-1m

python -m torch.distributed.launch --nproc_per_node=1 --use_env ncf.py --data /data/cache/ml-1m --epochs 1