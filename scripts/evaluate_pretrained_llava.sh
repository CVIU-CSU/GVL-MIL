#!/bin/bash
set -eo pipefail  # 启用严格错误检查
set -u            # 防止未声明变量

#######################################
### 全局配置（使用分组注释增强可读性） ###
#######################################
readonly PROJECT_BASE="/root/userfolder/MIL/VL-MIL"
readonly DATA_BASE="/root/userfolder/data-ckpts/VL-MIL"
readonly IMAGE_DIR="/root/commonfile/InfantVQA"
readonly BASE_MODEL_NAME="llava-onevision-qwen2-0.5b-si"
readonly MODEL_ROOT="/root/userfolder/model_weights"
# readonly BASE_MODEL_PATH="${}"
readonly VISION_TOWER_PATH=${MODEL_ROOT}/siglip-so400m-patch14-384
readonly PROMPT_VER="qwen_2"


##############################
### 路径配置（动态生成模板） ###
##############################
readonly RES_ROOT=${DATA_BASE}/checkpoints/v2/llava
readonly EVAL_LOG_DIR_TMPL=${RES_ROOT}/results/${BASE_MODEL_NAME}
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

########################
### 评估函数封装###
########################
run_evaluation() {
    mkdir -p "${EVAL_LOG_DIR_TMPL}"
    CUDA_VISIBLE_DEVICES=0 python ${PROJECT_BASE}/llava/eval/model_vqa.py \
        --num-chunks 1 \
        --model-path /root/userfolder/model_weights/llava-onevision-qwen2-7b-si \
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

    echo "===== 开始测试 ====="
    mkdir -p ${EVAL_LOG_DIR_TMPL}
    run_evaluation
    echo "===== 完成测试 ====="
}

main "$@"