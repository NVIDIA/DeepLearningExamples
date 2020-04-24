#!/bin/bash

BATCH_SIZE=1
INPUT_LENGTH=128
PRECISION="fp32"
NUM_ITERS=1003 # extra 3 iterations for warmup
TACOTRON2_CKPT="checkpoint_Tacotron2_1500_fp32"
WAVEGLOW_CKPT="checkpoint_WaveGlow_1000_fp32"
AMP_RUN=""
TEST_PROGRAM="test_infer.py"
WN_CHANNELS=512

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
	--test)
	    TEST_PROGRAM="$2"
	    shift
	    ;;
	--tacotron2)
	    TACOTRON2_CKPT="$2"
	    shift
	    ;;
	--encoder)
	    ENCODER_CKPT="$2"
	    shift
	    ;;
	--decoder)
	    DECODER_CKPT="$2"
	    shift
	    ;;
	--postnet)
	    POSTNET_CKPT="$2"
	    shift
	    ;;
	--waveglow)
	    WAVEGLOW_CKPT="$2"
	    shift
	    ;;
	--wn-channels)
	    WN_CHANNELS="$2"
	    shift
	    ;;
	*)
	    echo "Option $1 not recognized"
    esac
    shift
done

if [ "$PRECISION" = "amp" ]
then
    AMP_RUN="--amp-run"
elif  [ "$PRECISION" = "fp16" ]
then
    AMP_RUN="--fp16"
fi

LOG_SUFFIX=bs${BATCH_SIZE}_il${INPUT_LENGTH}_${PRECISION}
NVLOG_FILE=nvlog_${LOG_SUFFIX}.json
TMP_LOGFILE=tmp_log_${LOG_SUFFIX}.log
LOGFILE=log_${LOG_SUFFIX}.log


if [ "$TEST_PROGRAM" = "trt/test_infer_trt.py" ]
then
    TACOTRON2_PARAMS="--encoder $ENCODER_CKPT --decoder $DECODER_CKPT --postnet $POSTNET_CKPT"
else
    TACOTRON2_PARAMS="--tacotron2 $TACOTRON2_CKPT"
fi

set -x
python $TEST_PROGRAM \
       $TACOTRON2_PARAMS \
       --waveglow $WAVEGLOW_CKPT \
       --batch-size $BATCH_SIZE \
       --input-length $INPUT_LENGTH $AMP_RUN \
       --log-file $NVLOG_FILE \
       --num-iters $NUM_ITERS \
       --wn-channels $WN_CHANNELS \
       |& tee $TMP_LOGFILE
set +x


PERF=$(cat $TMP_LOGFILE | grep -F 'Throughput average (samples/sec)' | awk -F'= ' '{print $2}')
NUM_MELS=$(cat $TMP_LOGFILE | grep -F 'Number of mels per audio average' | awk -F'= ' '{print $2}')
LATENCY=$(cat $TMP_LOGFILE | grep -F 'Latency average (seconds)' | awk -F'= ' '{print $2}')
LATENCYSTD=$(cat $TMP_LOGFILE | grep -F 'Latency std (seconds)' | awk -F'= ' '{print $2}')
LATENCY90=$(cat $TMP_LOGFILE | grep -F 'Latency cl 90 (seconds)' | awk -F'= ' '{print $2}')
LATENCY95=$(cat $TMP_LOGFILE | grep -F 'Latency cl 95 (seconds)' | awk -F'= ' '{print $2}')
LATENCY99=$(cat $TMP_LOGFILE | grep -F 'Latency cl 99 (seconds)' | awk -F'= ' '{print $2}')

echo "$BATCH_SIZE,$INPUT_LENGTH,$PRECISION,$NUM_ITERS,$LATENCY,$LATENCYSTD,$LATENCY90,$LATENCY95,$LATENCY99,$PERF,$NUM_MELS" >> $LOGFILE
