import torch
import os 
import json
from PIL import Image

# 需要定义llava模块的位置，即指定PATH变量
import sys
llava_module_dir = "/root/userfolder/MIL/VL-MIL"
sys.path.insert(0, llava_module_dir)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.eval.cot_style_model_vqa import preprocess_qwen


def generate_hidden_states(
        line, 
        model,
        tokenizer,
        image_processor,
        return_encoder_output=True,
        image_folder="/root/commonfile/InfantVQA",
        device='cuda',
        system_message="You are a helpful assistant."
    ):
    """
    params:
        line: llava sample line
    return:
        1. SigLIP image_features: [num_patches, hidden_size]
        2. input_hidden_states: tuple of [batch, input_len, hidden_size] for each layer
        3. output_hidden_states: list of [num_gen_tokens, hidden_size], each item is one layer's outputs
    """
    # === 1. 解析输入 ===
    idx = line['id']
    image_files = line['image']
    # make sure image_files is a list of image
    if isinstance(image_files, str):
        image_files = [image_files]
    qs = line['conversations'][0]['value']
    conv_mode = 'qwen_2'

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize input (without generating)
    input_ids = preprocess_qwen(
        [line["conversations"][0], {'from': 'gpt', 'value': None}],
        tokenizer,
        has_image=True,
        system_message=system_message
    ).to(device)

    # === 2. 加载图像 ===
    image_tensors = []
    image_sizes = []
    modalities = []

    for image_file in image_files:
        if image_file.endswith(('.jpg', '.png')):
            modalities.append('image')
            image = Image.open(os.path.join(image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_sizes.append(image.size)  # 或固定值，根据模型需要
            image_tensors.append(image_tensor.half().to(device))

    # === 3. 获取 SigLIP 图像特征 ===
    if return_encoder_output:
        with torch.inference_mode():
            image_forward_outs = model.get_model().get_vision_tower()(image_tensors[0])
            # image_forward_outs: [1, num_patches, hidden_size]
            siglip_features = image_forward_outs.squeeze(0).cpu().detach()  # [num_patches, hidden_size]
    else:
        siglip_features = None

    # === 4. 多模态输入嵌入：获取 inputs_embeds 和 attention_mask ===
    with torch.inference_mode():
        (
            input_ids_multimodal,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None, None, None, None,
            image_tensors,
            modalities,
            image_sizes
        )

    input_len = inputs_embeds.shape[1]  # 输入序列长度（含 image tokens）

    # === 5. 获取输入序列的逐层 hidden states（Prefill 阶段）===
    with torch.inference_mode():
        outputs = model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True  # 为后续生成准备 KV Cache
        )

    # input_hidden_states: tuple of [1, input_len, hidden_size] for each layer
    input_hidden_states = tuple(hs.squeeze(0).cpu().detach() for hs in outputs.hidden_states)
    # 现在 shape: [input_len, hidden_size] per layer

    # === 6. 生成输出，并获取生成 token 的 hidden states ===
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        gen_outputs = model.generate(
            input_ids,
            images=image_tensors,
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            modalities=modalities,
            image_sizes=image_sizes,
            return_dict_in_generate=True,
            output_hidden_states=True,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    # === 7. 提取输出序列的 hidden states（每个生成 token 一个 step）===
    # gen_outputs.hidden_states: (step_0, step_1, ..., step_T)
    # step_0: prefill -> [1, input_len, h]
    # step_1+: 生成 -> [1, 1, h]

    num_layers = len(gen_outputs.hidden_states[0])  # 包括 embedding 层
    output_hidden_states = [[] for _ in range(num_layers)]  # 每层一个 list

    # 从 step=1 开始，step_0 是 prefill
    for step_idx in range(1, len(gen_outputs.hidden_states)):
        step_hidden = gen_outputs.hidden_states[step_idx]  # tuple of layer outputs
        for layer_idx in range(num_layers):
            # 取最后一个 token（当前生成的），去掉 batch 维度
            h = step_hidden[layer_idx][0, 0, :]  # [hidden_size]
            output_hidden_states[layer_idx].append(h.cpu().detach())

    # 拼接成 tensor: [num_gen_tokens, hidden_size] per layer
    output_hidden_states = [
        torch.stack(layer_hs) if len(layer_hs) > 0 else torch.empty(0, model.config.hidden_size)
        for layer_hs in output_hidden_states
    ]

    # === 8. 返回三部分 ===
    return {
        "num_layers": num_layers,                     # num of layers
        "siglip_features": siglip_features,           # [num_patches, hidden_size]
        "input_hidden_states": input_hidden_states,   # tuple of [input_len, hidden_size]
        "output_hidden_states": output_hidden_states  # list of [num_gen_tokens, hidden_size]
    }


def matrix_based_entropy(H: torch.Tensor) -> float:
    """
    H: hidden state [batch, L, D] (for single image: batch=1)
    """
    # Squeeze batch dim for single instance
    if H.dim() == 3:
        H = H.squeeze(0)  # [L, D]
    # 确保使用双精度提高数值稳定性
    if H.dtype != torch.float64:
        H = H.to(torch.float64)
    # 根据维度选择高效计算方法
    L, D = H.shape
    if L <= D:
        K = H @ H.t()  # [L, L]
    else:
        K = H.t() @ H  # [D, D] 更高效
    # 添加正则化项改善病态性
    regularizer = 1e-8 * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
    K += regularizer
    # 计算特征值
    eigvals = torch.linalg.eigvalsh(K)    
    # 过滤特征值 (使用更宽松的阈值)
    eigvals = eigvals[eigvals > 1e-10]
    # 计算熵
    p = eigvals / eigvals.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return entropy