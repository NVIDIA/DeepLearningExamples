#!/bin/bash

BATCH_SIZE=1
INPUT_LENGTH=128
PRECISION="fp32"
NUM_ITERS=1003 # extra 3 iterations for warmup
TACOTRON2_CKPT="checkpoint_Tacotron2_1500_fp32"
WAVEGLOW_CKPT="checkpoint_WaveGlow_1000_fp32"


while [ -n "$1" ]
do
    case "$1" in
	-bs|--batch-size)
	    BATCH_SIZE="$2"
	    shift
	    ;;
	-il|--input-length)
	    INPUT_LENGTH="$2"
	    shift
	    ;;
	-p|--prec)
	    PRECISION="$2"
	    shift
	    ;;
	--num-iters)
	    NUM_ITERS="$2"
	    shift
	    ;;
	--tacotron2)
	    TACOTRON2_CKPT="$2"
	    shift
	    ;;
	--waveglow)
	    WAVEGLOW_CKPT="$2"
	    shift
	    ;;
	*)
	    echo "Option $1 not recognized"
    esac
    shift
done

LOG_SUFFIX=bs${BATCH_SIZE}_il${INPUT_LENGTH}_${PRECISION}
NVLOG_FILE=nvlog_${LOG_SUFFIX}.json
TMP_LOGFILE=tmp_log_${LOG_SUFFIX}.log
LOGFILE=log_${LOG_SUFFIX}.log

set -x
python test_infer.py \
       --tacotron2 $TACOTRON2_CKPT \
       --waveglow $WAVEGLOW_CKPT \
       --batch-size $BATCH_SIZE \
       --input-length $INPUT_LENGTH $AMP_RUN $CPU_RUN \
       --log-file $NVLOG_FILE \
       --num-iters $NUM_ITERS \
       |& tee $TMP_LOGFILE
set +x


PERF=$(cat $TMP_LOGFILE | grep -F 'Throughput average (samples/sec)' | awk -F'= ' '{print $2}')
NUM_MELS=$(cat $TMP_LOGFILE | grep -F 'Number of mels per audio average' | awk -F'= ' '{print $2}')
LATENCY=$(cat $TMP_LOGFILE | grep -F 'Latency average (seconds)' | awk -F'= ' '{print $2}')
LATENCYSTD=$(cat $TMP_LOGFILE | grep -F 'Latency std (seconds)' | awk -F'= ' '{print $2}')
LATENCY50=$(cat $TMP_LOGFILE | grep -F 'Latency cl 50 (seconds)' | awk -F'= ' '{print $2}')
LATENCY100=$(cat $TMP_LOGFILE | grep -F 'Latency cl 100 (seconds)' | awk -F'= ' '{print $2}')

echo "$BATCH_SIZE,$INPUT_LENGTH,$PRECISION,$NUM_ITERS,$LATENCY,$LATENCYSTD,$LATENCY50,$LATENCY100,$PERF,$NUM_MELS" >> $LOGFILE
