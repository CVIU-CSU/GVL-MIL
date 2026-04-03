# collator and resampler
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import Sampler
import numpy as np


def batch_collate_fn(batch):
    # valid batch
    batch = [b for b in batch if b is not None]
    if not batch:
        return None 
    # load feature
    ids = [b['id'] for b in batch]
    features = [b["features"] for b in batch]
    # if output sequence length is not same
    D_max = max(f.shape[1] for f in features)
    padded_D_features = []
    for f in features:
        if f.shape[1] < D_max:
            pad_size = D_max - f.shape[1]
            f = torch.nn.functional.pad(f, (0, 0, 0, pad_size))  # 在最后一维 padding
        padded_D_features.append(f)
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    instance_nums = [f.shape[0] for f in features]
    # pad features
    padded_features = pad_sequence(padded_D_features, batch_first=True).contiguous()
    # padded_features = pad_sequence(features, batch_first=True).contiguous()
    # try:
    #     padded_features = pad_sequence(features, batch_first=True)
    # except:
    #     print([f.shape for f in features])
    #     padded_features = pad_sequence(features, batch_first=True)
        
    # masks: [batch_size, M_instance_num]
    max_K = padded_features.shape[1]
    device = padded_features.device
    arange_K = torch.arange(max_K, device=device).unsqueeze(0)
    instance_lens = torch.tensor(instance_nums, device=device).unsqueeze(1)  # [B,1]
    masks = (arange_K < instance_lens).float()  # broadcasting [B, max_K]
    return {
        'ids': ids, 
        "features": padded_features.float(),
        "labels": labels,
        "instance_num": instance_nums,
        "masks": masks
    }

def last_token(features):
    is_valid = features.abs().sum(dim=-1) > 1e-10
    last_valid_idx = is_valid.sum(dim=1) - 1
    bag_indices = torch.arange(features.size(0))
    return features[bag_indices, last_valid_idx]

def dual_modal_batch_collate_fn(batch):
    # valid batch
    batch = [b for b in batch if b is not None]
    if not batch:
        return None 
    # load feature
    ids = [b['id'] for b in batch]
    input_features = [b["input_features"] for b in batch]
    output_features = [last_token(b["output_features"]) for b in batch]
    
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    instance_nums = [f.shape[0] for f in input_features]
    # pad features
    padded_input_features = pad_sequence(input_features, batch_first=True).contiguous()
    padded_output_features = pad_sequence(output_features, batch_first=True).contiguous()

    max_K = padded_output_features.shape[1]
    device = padded_output_features.device
    arange_K = torch.arange(max_K, device=device).unsqueeze(0)
    instance_lens = torch.tensor(instance_nums, device=device).unsqueeze(1)  # [B,1]
    masks = (arange_K < instance_lens).float()  # broadcasting [B, max_K]
    return {
        'ids': ids, 
        "input_features": padded_input_features.float(),
        "output_features": padded_output_features.float(),
        "labels": labels,
        "instance_num": instance_nums,
        "masks": masks
    }

class NeonatalFundusResampler(Sampler):
    def __init__(self, dataset, select_ratio=0.25, shuffle=True, seed=None):
        self.dataset = dataset
        self.select_ratio = select_ratio
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.norm_indices = []
        self.abnorm_indices = []

        for idx, sample_id in enumerate(dataset.get_ids()):
            if sample_id // 10000 == 0:
                self.norm_indices.append(idx)
            else:
                self.abnorm_indices.append(idx)
        print(f"[Resampler] {len(self.norm_indices)} normal samples, {len(self.abnorm_indices)} abnormal samples.")        
        self.norm_indices = np.array(self.norm_indices)
        self.abnorm_indices = np.array(self.abnorm_indices)
        # set random seed


    def __iter__(self):
        # if self.seed is not None:
        #     np.random.seed(self.seed + self.epoch)
        self.epoch += 1
        
        selected_norm = np.random.choice(
            self.norm_indices,
            size=int(len(self.norm_indices) * self.select_ratio),
            replace=False
        )
        
        indices = np.concatenate([selected_norm, self.abnorm_indices])
        
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        return iter(indices.tolist())
    
    def __len__(self):
        return int(len(self.norm_indices) * self.select_ratio) + len(self.abnorm_indices)

