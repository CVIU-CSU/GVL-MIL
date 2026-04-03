#!/bin/bash
cd /root/userfolder/MIL/VL-MIL/

train_mil_one(){
    local mil_type=$1
    local mil_lr=$2
    local agg_lr=$3
    local batch_size=$4
    local grad_acc_step=$5
    local layer=$6
    local idx=$7
    CUDA_VISIBLE_DEVICES=0 python  mil/gvlmil_main.py \
        --mil-name $mil_type --mil-lr $mil_lr --resample-ratio 0.25 \
        --batch-size $batch_size --grad-acc-step $grad_acc_step --num-epochs 20 \
        --exp-idx $idx --exp-name main_ours \
        --encoder qwen2-7b --tokens image --layer $layer --test
}


train_mil_one "dsmil" 2e-5 2e-5 4 16 23 0
