#!/bin/bash
set -eo pipefail  # 启用严格错误检查
set -u            # 防止未声明变量

#######################################
### 全局配置（使用分组注释增强可读性） ###
#######################################
readonly PROJECT_BASE="/root/userfolder/MIL/VL-MIL"
readonly DATA_BASE="/root/userfolder/data-ckpts/VL-MIL"
readonly IMAGE_DIR="/root/commonfile/InfantVQA"
readonly BASE_MODEL_NAME="llava-onevision-qwen2-7b-si"
readonly MODEL_ROOT="/root/userfolder/model_weights"
# readonly BASE_MODEL_PATH="${}"
readonly VISION_TOWER_PATH=${MODEL_ROOT}/siglip-so400m-patch14-384
readonly PROMPT_VER="qwen_2"
readonly NUM_EPOCHS=3
################################
### 训练参数（分组展示关键参数） ###
################################
readonly LORA_RANK=32
readonly LORA_ALPHA=128
readonly BASE_LR=1e-4
readonly BATCH_SIZE=1
readonly GRAD_ACCUM_STEPS=8
readonly MAX_GRID_NUM="3x3"
readonly NUM_TRY="freeze_backbone_valid_desc_only"

##############################
### 路径配置（动态生成模板） ###
##############################
readonly RES_ROOT=${DATA_BASE}/checkpoints/v2/llava
readonly CHECKPOINT_DIR_TMPL=${RES_ROOT}/ckpt/${BASE_MODEL_NAME}-ft-lora-${NUM_EPOCHS}epoch-NFD-${NUM_TRY}
readonly MERGED_DIR_TMPL=${RES_ROOT}/merged/${BASE_MODEL_NAME}-ft-${NUM_EPOCHS}epoch-NFD-${NUM_TRY}
readonly EVAL_LOG_DIR_TMPL=${RES_ROOT}/results/${BASE_MODEL_NAME}-ft-lora-${NUM_EPOCHS}epoch-NFD-${NUM_TRY}
# readonly TEST_LOG_IIR_TMPL="${RES_ROOT}/results/test/llavaov${NUM_TRY}/${BASE_MODEL_NAME}-ft-lora-{fold}-${NUM_EPOCHS}epoch-InfantVQA${NUM_TRY}"

####################################
### 初始化验证（避免路径错误）###
####################################
validate_paths() {
    local required_dirs=("${PROJECT_BASE}" "${IMAGE_DIR}" "${MODEL_ROOT}")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "${dir}" ]]; then
            echo "[ERROR] 关键目录不存在: ${dir}" >&2
            exit 1
        fi
    done
}

cd ${PROJECT_BASE}

####################
### 模型训练函数封装 ###
####################
train_model() {
    mkdir -p ${CHECKPOINT_DIR_TMPL}
    deepspeed --include="localhost:1" --master_port=29500 \
        ${PROJECT_BASE}/llava/train/train.py \
        --lora_enable True \
        --lora_r ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --deepspeed ${PROJECT_BASE}/scripts/zero_config/zero2.json \
        --model_name_or_path ${MODEL_ROOT}/${BASE_MODEL_NAME} \
        --version ${PROMPT_VER} \
        --data_path ${DATA_BASE}/datasets/v2/description_only/nfd_train.json \
        --image_folder ${IMAGE_DIR} \
        --vision_tower ${VISION_TOWER_PATH} \
        --mm_tunable_parts mm_mlp_adapter,mm_language_model \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --group_by_modality_length True \
        --image_aspect_ratio anyres \
        --image_grid_pinpoints "(1x1),...,(${MAX_GRID_NUM})" \
        --mm_patch_merge_type spatial_unpad \
        --bf16 True \
        --output_dir ${CHECKPOINT_DIR_TMPL} \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
        --learning_rate ${BASE_LR} \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 3000 \
        --save_total_limit 1 \
        --learning_rate ${BASE_LR} \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 32768 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --torch_compile True \
        --torch_compile_backend "inductor" \
        --dataloader_drop_last True \
        --report_to wandb \
        --freeze_backbone True \
        2>&1 | tee ${CHECKPOINT_DIR_TMPL}/train.log
}

########################
### 权重合并函数封装[7](@ref) ###
########################
merge_weights() {
    mkdir -p ${MERGED_DIR_TMPL}
    CUDA_VISIBLE_DEVICES=0 python ${PROJECT_BASE}/llava/merge_lora_weights.py \
        --model-path ${CHECKPOINT_DIR_TMPL} \
        --model-base ${MODEL_ROOT}/${BASE_MODEL_NAME} \
        --save-model-path ${MERGED_DIR_TMPL} \
        2>&1 | tee ${MERGED_DIR_TMPL}/merge.log
}

########################
### 评估函数封装###
########################
run_evaluation() {
    echo "Load model from ${MERGED_DIR_TMPL}"

    mkdir -p "${EVAL_LOG_DIR_TMPL}"
    CUDA_VISIBLE_DEVICES=0 python ${PROJECT_BASE}/llava/eval/model_vqa.py \
        --num-chunks 1 \
        --model-path ${MERGED_DIR_TMPL} \
        --question-file ${DATA_BASE}/datasets/v2/description_only/nfd_valid.json \
        --image-folder ${IMAGE_DIR} \
        --answers-file ${EVAL_LOG_DIR_TMPL}/NFD_answers.jsonl \
        2>&1 | tee ${EVAL_LOG_DIR_TMPL}/eval_NFD.log

    echo "Evaluation results at ${EVAL_LOG_DIR_TMPL}"
}

#################
### 主执行流程 ###
#################
main() {
    validate_paths  # 前置检查

    cd ${PROJECT_BASE}
    
    echo "===== 开始训练 ====="
    # mkdir -p ${CHECKPOINT_DIR_TMPL}
    # train_model
    echo "===== 完成训练 ====="
    
    echo "===== 开始合并 ====="
    # mkdir -p ${MERGED_DIR_TMPL}
    # merge_weights
    echo "===== 完成合并 ====="

    echo "===== 开始测试 ====="
    mkdir -p ${EVAL_LOG_DIR_TMPL}
    run_evaluation
    echo "===== 完成测试 ====="
}

main "$@"