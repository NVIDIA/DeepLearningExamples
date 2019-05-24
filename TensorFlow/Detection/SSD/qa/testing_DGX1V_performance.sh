if [[ -z $CMD || -z $BASELINE || -z $TOLERANCE ]]
then
    echo some variables is not set
    exit 1
fi


echo $MSG
RESULT=$($CMD)

imgps=$(echo $RESULT | tail -n 1 | awk '{print $3}')
LB_imgps=$(python -c "print($BASELINE * (1-$TOLERANCE))")

echo imgs/s: $imgps expected imgs/s: $BASELINE
echo accepted minimum: $LB_imgps

if [[ $imgps > $LB_imgps ]]
then
    echo PASSED
else
    echo $RESULT
    echo FAILED
    exit 1
fi
