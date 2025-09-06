#!/bin/bash
mkdir -p logs
killall -q find-optimal

START=`date +%s%N | cut -b1-13`
CUDA_VISIBLE_DEVICES=0 ./build/find-optimal $1 -b1000000000001 -e2000000000000 > logs/gpu1.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=1 ./build/find-optimal $1 -b2000000000001 -e3000000000000 > logs/gpu2.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=2 ./build/find-optimal $1 -b3000000000001 -e4000000000000 > logs/gpu3.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=3 ./build/find-optimal $1 -b4000000000001 -e5000000000000 > logs/gpu4.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=4 ./build/find-optimal $1 -b5000000000001 -e6000000000000 > logs/gpu5.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=5 ./build/find-optimal $1 -b6000000000001 -e7000000000000 > logs/gpu6.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=6 ./build/find-optimal $1 -b7000000000001 -e8000000000000 > logs/gpu7.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=7 ./build/find-optimal $1 -b8000000000001 -e9000000000000 > logs/gpu8.log &
echo "Launched: $!"

# Wait on everything
for job in `jobs -p`; do echo "Waiting on $job"; wait $job; done
END=`date +%s%N | cut -b1-13`
PERSEC=`echo "$(( 8000000000000 / ($END-$START) * 1000 ))"`
printf "%'d per second\n" $PERSEC
