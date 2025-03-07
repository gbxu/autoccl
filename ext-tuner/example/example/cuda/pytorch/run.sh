#!/bin/bash
set -x
THIS_PATH=$(readlink -f "$0")
THIS_DIR=$(dirname "$THIS_PATH")
NCCL_HOME="$THIS_DIR/../../../../../build"
TUNER_HOME="$THIS_DIR/../../../build"

export TUNER_MAXCHANNELS=32
export TUNER_P2P_NCHANNELS=2
export TUNER_WHITELIST_CASES_FILE="./whitelistcases.txt"
export TUNER_WHITELIST_RULES_FILE="./whitelistrules.txt"
export NCCL_TIMEOUT=3600
export TUNER_PRETRAIN_STEPS=360
export TUNER_TRAIN_STEPS=240
export TUNER_PROFILE_REPEAT=5
export TUNER_COORDINATOR=localhost:12449
export TUNER_WORLDSIZE=8
export NCCL_TUNER_PLUGIN=${TUNER_HOME}/libnccl-plugin.so
export LD_PRELOAD=${NCCL_HOME}/lib/libnccl.so:$LD_PRELOAD
export LD_LIBRARY_PATH=${NCCL_HOME}/lib:${TUNER_HOME}:$LD_LIBRARY_PATH

torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 demo.py
