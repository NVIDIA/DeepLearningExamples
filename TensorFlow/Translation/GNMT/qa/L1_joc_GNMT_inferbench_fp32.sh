set -o nounset
set -o errexit
set -o pipefail

cd .. 
cp -r /data/joc/gnmt_tf/19.08 output_dir

# hack to work with pytorch dataset
sed -ie 's/    src_vocab_file = hparams.vocab_prefix + "." + hparams.src/    src_vocab_file = hparams.vocab_prefix/g' nmt.py
sed -ie 's/    tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt/    tgt_vocab_file = hparams.vocab_prefix/g' nmt.py

( python nmt.py --data_dir=/data/pytorch/wmt16_de_en --output_dir=output_dir --mode=infer --infer_batch_size=512 2>&1 ) | tee log.log
python scripts/parse_log.py log.log | tee log.json

python << END
import json
import numpy as np
from pathlib import Path

baseline = 5374
bleu_baseline = 25.1

log = json.loads(Path('log.json').read_text())
speed = np.mean(log['eval_tokens_per_sec'])
bleu = log['bleu'][0]

print('Eval speed    :', speed)
print('Baseline      :', baseline)

print('Bleu          :', bleu)
print('Bleu baseline :', bleu_baseline)

if speed < baseline * 0.9:
    print("FAILED: speed ({}) doesn't match the baseline ({})".format(speed, baseline))
    exit(1)

if bleu < bleu_baseline - 0.2:
    print("FAILED: bleu ({}) doesn't match the baseline ({})".format(bleu, bleu_baseline))
    exit(1)

print('SUCCESS')
END

