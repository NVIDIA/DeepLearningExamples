#!/bin/bash

echo "Processing file $1"
echo "Throughput:"

cat $1 | cut -d ',' -f3

echo ""
echo "Average latency: "
cat $1 | cut -d ',' -f4-10 | sed "s/,/\+/g" | sed "s/.*/scale=2; (\0) \/ 1000/g" | bc

echo ""
echo "p90 latency: "
cat $1 | cut -d ',' -f12 | sed "s/.*/scale=2; \0 \/ 1000/g" | bc
echo ""
echo "p95 latency: "
cat $1 | cut -d ',' -f13 | sed "s/.*/scale=2; \0 \/ 1000/g" | bc
echo ""
echo "p99 latency: "
cat $1 | cut -d ',' -f14 | sed "s/.*/scale=2; \0 \/ 1000/g" | bc