#! /bin/bash
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node=1 test_attention.py