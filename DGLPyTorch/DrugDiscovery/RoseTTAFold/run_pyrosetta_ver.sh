#!/bin/bash

# make the script stop when error (non-true exit code) is occured
set -e

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################

SCRIPT=`realpath -s $0`
export PIPEDIR=`dirname $SCRIPT`

CPU="8"  # number of CPUs to use
MEM="64" # max memory (in GB)

# Inputs:
IN="$1"                # input.fasta
WDIR=`realpath -s $2`  # working folder


LEN=`tail -n1 $IN | wc -m`

mkdir -p $WDIR/log

conda activate RoseTTAFold
############################################################
# 1. generate MSAs
############################################################
if [ ! -s $WDIR/t000_.msa0.a3m ]
then
    echo "Running HHblits"
    $PIPEDIR/input_prep/make_msa.sh $IN $WDIR $CPU $MEM > $WDIR/log/make_msa.stdout 2> $WDIR/log/make_msa.stderr
fi


############################################################
# 2. predict secondary structure for HHsearch run
############################################################
if [ ! -s $WDIR/t000_.ss2 ]
then
    echo "Running PSIPRED"
    $PIPEDIR/input_prep/make_ss.sh $WDIR/t000_.msa0.a3m $WDIR/t000_.ss2 > $WDIR/log/make_ss.stdout 2> $WDIR/log/make_ss.stderr
fi


############################################################
# 3. search for templates
############################################################
DB="$PIPEDIR/pdb100_2021Mar03/pdb100_2021Mar03"
if [ ! -s $WDIR/t000_.hhr ]
then
    echo "Running hhsearch"
    HH="hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 100 -p 5.0 -d $DB"
    cat $WDIR/t000_.ss2 $WDIR/t000_.msa0.a3m > $WDIR/t000_.msa0.ss2.a3m
    $HH -i $WDIR/t000_.msa0.ss2.a3m -o $WDIR/t000_.hhr -atab $WDIR/t000_.atab -v 0 > $WDIR/log/hhsearch.stdout 2> $WDIR/log/hhsearch.stderr
fi


############################################################
# 4. predict distances and orientations
############################################################
if [ ! -s $WDIR/t000_.3track.npz ]
then
    echo "Predicting distance and orientations"
    python $PIPEDIR/network/predict_pyRosetta.py \
        -m $PIPEDIR/weights \
        -i $WDIR/t000_.msa0.a3m \
        -o $WDIR/t000_.3track \
        --hhr $WDIR/t000_.hhr \
        --atab $WDIR/t000_.atab \
        --db $DB 1> $WDIR/log/network.stdout 2> $WDIR/log/network.stderr
fi

############################################################
# 5. perform modeling
############################################################
mkdir -p $WDIR/pdb-3track

conda deactivate
conda activate folding

for m in 0 1 2
do
    for p in 0.05 0.15 0.25 0.35 0.45
    do
        for ((i=0;i<1;i++))
        do
            if [ ! -f $WDIR/pdb-3track/model${i}_${m}_${p}.pdb ]; then
                echo "python -u $PIPEDIR/folding/RosettaTR.py --roll -r 3 -pd $p -m $m -sg 7,3 $WDIR/t000_.3track.npz $IN $WDIR/pdb-3track/model${i}_${m}_${p}.pdb"
            fi
        done
    done
done > $WDIR/parallel.fold.list

N=`cat $WDIR/parallel.fold.list | wc -l`
if [ "$N" -gt "0" ]; then
    echo "Running parallel RosettaTR.py"    
    parallel -j $CPU < $WDIR/parallel.fold.list > $WDIR/log/folding.stdout 2> $WDIR/log/folding.stderr
fi

############################################################
# 6. Pick final models
############################################################
count=$(find $WDIR/pdb-3track -maxdepth 1 -name '*.npz' | grep -v 'features' | wc -l)
if [ "$count" -lt "15" ]; then
    # run DeepAccNet-msa
    echo "Running DeepAccNet-msa"
    python $PIPEDIR/DAN-msa/ErrorPredictorMSA.py --roll -p $CPU $WDIR/t000_.3track.npz $WDIR/pdb-3track $WDIR/pdb-3track 1> $WDIR/log/DAN_msa.stdout 2> $WDIR/log/DAN_msa.stderr
fi

if [ ! -s $WDIR/model/model_5.crderr.pdb ]
then
    echo "Picking final models"
    python -u -W ignore $PIPEDIR/DAN-msa/pick_final_models.div.py \
        $WDIR/pdb-3track $WDIR/model $CPU > $WDIR/log/pick.stdout 2> $WDIR/log/pick.stderr
    echo "Final models saved in: $2/model"
fi
echo "Done"
