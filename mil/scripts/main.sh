#!/bin/bash
BACKBONE="siglip"

cd /root/userfolder/MIL/VL-MIL/

train_mil() {
    local mil_type=$1
    local mil_lr=$2
    local batch_size=$3
    for idx in {0..4}; do
        python  mil/main.py \
            --mil-name $mil_type --mil-lr $mil_lr --resample-ratio 0.25 \
            --batch-size $batch_size --num-epochs 100 --exp-idx $idx --test 
    done
}

train_mil_one(){
    local mil_type=$1
    local mil_lr=$2
    local batch_size=$3
    local idx=$4
    python  mil/main.py \
        --mil-name $mil_type --mil-lr $mil_lr --resample-ratio 0.25 \
        --batch-size $batch_size --num-epochs 100 --exp-idx $idx --test
}

train_mil "clam" "2e-5" 64
train_mil "dftd" "2e-5" 64
train_mil "abmil" "2e-4" 64
train_mil "dsmil" "2e-4" 64
train_mil "ilra" "1e-4" 64
train_mil "rrt" "1.5e-4" 64
train_mil "transmil" "1e-5" 64
train_mil "transformer" "2e-5" 64
train_mil "wikg" "1e-4" 64

# train_mil_one "clam" 2e-5 1 0
# train_mil_one "dftd" 2e-5 1 0
# train_mil_one "rrt" 1.5e-4 64 0
# train_mil_one "transformer" 2e-5 0
# train_mil "transmil" 1.5e-5 64
# train_mil_one "transformer" 2e-5 64 0
