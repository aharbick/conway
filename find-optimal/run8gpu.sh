#!/bin/bash
mkdir -p logs
killall -q find-optimal
CUDA_VISIBLE_DEVICES=0 ./build/find-optimal $1 -b1 -e2305843009213693951 > logs/gpu1.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=1 ./build/find-optimal $1 -b2305843009213693952 -e4611686018427387902 > logs/gpu2.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=2 ./build/find-optimal $1 -b4611686018427387903 -e6917529027641081853 > logs/gpu3.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=3 ./build/find-optimal $1 -b6917529027641081854 -e9223372036854775804 > logs/gpu4.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=4 ./build/find-optimal $1 -b9223372036854775805 -e11529215046068469755 > logs/gpu5.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=5 ./build/find-optimal $1 -b11529215046068469756 -e13835058055282163706 > logs/gpu6.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=6 ./build/find-optimal $1 -b13835058055282163707 -e16140901064495857657 > logs/gpu7.log &
echo "Launched: $!"
if [ "X$1" != "X" ]; then
  sleep 1
fi
CUDA_VISIBLE_DEVICES=7 ./build/find-optimal $1 -b16140901064495857658 -e18446744073709551608 > logs/gpu8.log &
echo "Launched: $!"

# Wait on everything
for job in `jobs -p`; do echo "Waiting on $job"; wait $job; done
