from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sys
mil_module_dir = "/root/userfolder/MIL/VL-MIL"
sys.path.insert(0, mil_module_dir)

from mil.models.abmil import ABMIL
from mil.models.clam import CLAMSB
from mil.models.dftd import DFTD
from mil.models.dsmil import DSMIL 
from mil.models.ilra import ILRA
from mil.models.rrt import RRTMIL
from mil.models.transmil import TransMIL
from mil.models.transformer import Transformer
from mil.models.wikg import WIKGMIL

def build_mil_model(
        model_name: str,
        model_config: str,
        pretrained: bool,
        encoder: str,
        num_classes: int,
        pretrained_cfg: Optional[Dict[str, Any]] = None,
        from_pretrained: bool = False,
        pretrained_strict: bool = False,
        keep_classifier: bool = False,
        **kwargs
):
    

if __name__ == "__main__":
    print("OK?")