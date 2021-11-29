#!/bin/bash
mkdir -p logs
killall find-cuda-optimal
CUDA_VISIBLE_DEVICES=0 ./build/find-cuda-optimal -g1 -s1 -e2305843009213693951 > logs/gpu1.log &
CUDA_VISIBLE_DEVICES=1 ./build/find-cuda-optimal -g1 -s2305843009213693952 -e4611686018427387902 > logs/gpu2.log &
CUDA_VISIBLE_DEVICES=2 ./build/find-cuda-optimal -g1 -s4611686018427387903 -e6917529027641081853 > logs/gpu3.log &
CUDA_VISIBLE_DEVICES=3 ./build/find-cuda-optimal -g1 -s6917529027641081854 -e9223372036854775804 > logs/gpu4.log &
CUDA_VISIBLE_DEVICES=4 ./build/find-cuda-optimal -g1 -s9223372036854775805 -e11529215046068469755 > logs/gpu5.log &
CUDA_VISIBLE_DEVICES=5 ./build/find-cuda-optimal -g1 -s11529215046068469756 -e13835058055282163706 > logs/gpu6.log &
CUDA_VISIBLE_DEVICES=6 ./build/find-cuda-optimal -g1 -s13835058055282163707 -e16140901064495857657 > logs/gpu7.log &
CUDA_VISIBLE_DEVICES=7 ./build/find-cuda-optimal -g1 -s16140901064495857658 -e18446744073709551608 > logs/gpu8.log &
