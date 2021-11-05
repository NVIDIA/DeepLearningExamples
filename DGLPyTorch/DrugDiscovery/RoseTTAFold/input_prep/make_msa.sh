#!/bin/bash

# inputs
in_fasta="$1"
out_dir="$2"

# resources
CPU="$3"
MEM="$4"

# sequence database
DB="$5/UniRef30_2020_06/UniRef30_2020_06"

# setup hhblits command
HHBLITS="hhblits -o /dev/null -mact 0.35 -maxfilt 100000000 -neffmax 20 -cov 25 -cpu $CPU -nodiff -realign_max 100000000 -maxseq 1000000 -maxmem $MEM -n 4 -d $DB"
echo $HHBLITS

mkdir -p $out_dir/hhblits
tmp_dir="$out_dir/hhblits"
out_prefix="$out_dir/t000_"

# perform iterative searches
prev_a3m="$in_fasta"
for e in 1e-30 1e-10 1e-6 1e-3
do
    echo $e
    $HHBLITS -i $prev_a3m -oa3m $tmp_dir/t000_.$e.a3m -e $e -v 0
    hhfilter -id 90 -cov 75 -i $tmp_dir/t000_.$e.a3m -o $tmp_dir/t000_.$e.id90cov75.a3m
    hhfilter -id 90 -cov 50 -i $tmp_dir/t000_.$e.a3m -o $tmp_dir/t000_.$e.id90cov50.a3m
    prev_a3m="$tmp_dir/t000_.$e.id90cov50.a3m"
    n75=`grep -c "^>" $tmp_dir/t000_.$e.id90cov75.a3m`
    n50=`grep -c "^>" $tmp_dir/t000_.$e.id90cov50.a3m`

    if ((n75>2000))
    then
        if [ ! -s ${out_prefix}.msa0.a3m ]
        then
            cp $tmp_dir/t000_.$e.id90cov75.a3m ${out_prefix}.msa0.a3m
	    break
        fi
    elif ((n50>4000))
    then
        if [ ! -s ${out_prefix}.msa0.a3m ]
        then
            cp $tmp_dir/t000_.$e.id90cov50.a3m ${out_prefix}.msa0.a3m
            break
        fi
    else
        continue
    fi

done

if [ ! -s ${out_prefix}.msa0.a3m ]
then
    cp $tmp_dir/t000_.1e-3.id90cov50.a3m ${out_prefix}.msa0.a3m
fi
