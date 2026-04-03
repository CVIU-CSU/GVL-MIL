import sys
mil_module_dir = "/root/userfolder/MIL/VL-MIL"
sys.path.insert(0, mil_module_dir)

from mil.models import (
    abmil, transmil, transformer, 
    ilra, dftd, clam, rrt, 
    wikg, dsmil
)
from mil.components.cross_attn_aggregator import XattnAggregator
from mil.components.ld2g import LD2GBlock
import pathlib
import os

# image tokens are input_tokens[IMAGE_POS[0]: IMAGE_POS[1]]
IMAGE_POS = (14, 743)
# instruct tokens are input_tokens[INSTRUCT_POS[0]: INSTRUCT_POS[1]]
INSTRUCT_POS = (746, 763)
# prompt + image + instruct
INPUT_LEN = 767

# label_dict
LABEL_DICT = {
    "未见明显异常。": 0,
    "视网膜出血。": 1,
    "早产儿视网膜病变。": 2,
    "白斑。": 3
}


REPO_PATH = str(pathlib.Path(__file__).parent.resolve())  # absolute path to repo root
CKPT_PATH = '/root/userfolder/data-ckpts/VL-MIL/checkpoints/v2'
CONFIG_PATH = os.path.join(REPO_PATH, 'model_configs')
MODEL_SAVE_PATH = os.path.join(CKPT_PATH, 'mil_weights')

ENCODER_DIM_MAPPING : dict[str, int] = {
    'qwen2-7b': 3584,
    'siglip': 1152,
    'qwen2-0.5b': 896
}

AGG_MAPPING ={
    'xattn': XattnAggregator,
    'ld2g': LD2GBlock
}

MODEL_ENTRYPOINTS = {
    'abmil': (abmil.ABMILModel, abmil.ABMILGatedBaseConfig),
    'transmil': (transmil.TransMILModel, transmil.TransMILConfig),
    'transformer': (transformer.TransformerModel, transformer.TransformerConfig),
    'dftd': (dftd.DFTDModel, dftd.DFTDConfig),
    'clam': (clam.CLAMModel, clam.CLAMConfig),
    'ilra': (ilra.ILRAModel, ilra.ILRAConfig),
    'rrt': (rrt.RRTMILModel, rrt.RRTMILConfig),
    'wikg': (wikg.WIKGMILModel, wikg.WIKGConfig),
    'dsmil': (dsmil.DSMILModel, dsmil.DSMILConfig),
}  

if __name__ == "__main__":
    print(f"REPO_PATH: {REPO_PATH}")
    print(f"CONFIG_PATH: {CONFIG_PATH}")
    print(f"MODEL_SAVE_PATH: {MODEL_SAVE_PATH}")
    os.makedirs(CONFIG_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)