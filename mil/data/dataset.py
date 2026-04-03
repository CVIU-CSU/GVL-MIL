# dataset
import json
import os 
from typing import Optional, Union, List, Tuple
import torch
from torch.utils.data import Dataset

from mil.constants import IMAGE_POS, INSTRUCT_POS, LABEL_DICT, ENCODER_DIM_MAPPING


class NeonatalFundusDataset(Dataset):
    def __init__(
            self,
            json_path: Union[str, List[str], Tuple[str]],
            feature_folder: str = "/root/commonfile/InfantVQA/nfi/features",
            split: str = "valid", # valid, test
            feature_type: str = "qwen", # siglip, qwen, dual
            layer: Optional[int] = None, # For SigLip feature, layer is None; For Qwen feature, layer represents the output from corresponding layer.
            tokens: Optional[str] = 'image' #image, instruct, input, output for qwen; output, pooler for siglip
    ):
        if isinstance(json_path, str):
            json_path = [json_path]
        elif isinstance(json_path, tuple):
            json_path = list(json_path)

        self.data = []
        # Read json files
        for path in json_path:
            print(f"[Dataset] Loading dataset from {path}")
            with open(path, 'r') as f:
                data = json.load(f)
            self.data.extend(
                {key: value for key, value in sample.items() if key not in ["image", "conversations"]}
                for sample in data
            )
            print(f"[Dataset] Successfully loaded {len(data)} samples from {path}")
        # for sample in data:
        #     self.data.append({key: value for key, value in sample.items() if key not in ["image", "conversations"]})
        self.feature_folder = feature_folder
        self.feature_type = feature_type
        self.split = split
        self.layer = layer 
        self.tokens = tokens 

    # return ids for sample selection
    def get_ids(self):
        return [sample['id'] for sample in self.data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index].copy()
        patient = sample['patient'].split('/')[-1]
        if "qwen" in self.feature_type:
            if self.layer is None or "output" in self.tokens:
                feature_name = "qwen_output_" + self.split
                feature_path = os.path.join(self.feature_folder, feature_name, f"{patient}.pt")
                features = torch.load(feature_path, map_location='cpu', mmap=True)

                if 'last' in self.tokens:
                    features = features[:, -1, :].unsqueeze(-2)
                # print(features.shape)
            else:
                depth = str(self.layer) if self.layer >= 10 else f"0{self.layer}"
                feature_name = f"qwen_layer_{depth}_{self.split}"
                feature_path = os.path.join(self.feature_folder, feature_name, f"{patient}.pt")
                features = torch.load(feature_path, map_location='cpu', mmap=True)
                if features.ndim == 2:
                    features = features.unsqueeze(0)
                # select tokens
                if 'image' in self.tokens or 'img' in self.tokens:
                    features = features[:, IMAGE_POS[0]: IMAGE_POS[1], :]
                elif 'text' in self.tokens or 'instruct' in self.tokens:
                    features = features[:, INSTRUCT_POS[0]: INSTRUCT_POS[1], :]
                elif 'last' in self.tokens:
                    features = features[:, -1, :].unsqueeze(-2)
                else: # neither image nor instruct, the whole input sequence
                    self.tokens = 'input'

        elif "siglip" in self.feature_type:
            if "pooler" in self.tokens:
                feature_name = "siglip_pooler"
            else:
                feature_name = "siglip_output"
            # load feature
            feature_path = os.path.join(self.feature_folder, feature_name, f"{patient}.pt")
            features = torch.load(feature_path, map_location='cpu', mmap=True)
        else:
            raise ValueError(f"Unrecognized feature type {self.feature_type}")
        
        sample['features'] = features 
        # if feature tensor's shape is [L, D], add K
        if sample['features'].ndim == 2:
            sample['features'] = sample['features'].unsqueeze(0)
        sample['label'] = LABEL_DICT[sample["label"]]
        return sample