import gc
import os
import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import json
import csv
# new
import sys
sys.path.insert(0, "/root/userfolder/MIL/VL-MIL")

from mil.builder import create_model, save_model, create_aggregator

from mil.data.dataset import NeonatalFundusDataset
from mil.data.data_components import batch_collate_fn, NeonatalFundusResampler
from mil.losses import LOSS_DICT
from mil.constants import MODEL_SAVE_PATH
from mil.train.utils import (
    visualize_errors, 
    print_metrics,
    print_label_metrics, 
    specificity_precision_recall_f1_auc_acc
)

def calculate_score(metrics):
    score = 0.6 * metrics['AUC'] + 0.3 * metrics['Sensitivity'] + 0.1 * metrics['F1']
    return score

class MILTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.loss_fn = LOSS_DICT[self.config.loss].to(self.device)
        # create ckpt save path
        self.ckpt_save_path = os.path.join(MODEL_SAVE_PATH, self.config.exp_name, f"{self.config.mil_name}-{self.config.exp_idx}")
        print(f"[Trainer] Checkpoints would be saved at \n{self.ckpt_save_path}")
        assert not os.path.exists(self.ckpt_save_path), f"{self.ckpt_save_path} exists."
        os.makedirs(self.ckpt_save_path, exist_ok=True)
        # save config
        config_dict = vars(config)
        with open(os.path.join(self.ckpt_save_path, 'training_args.json'), 'w') as file:
            json.dump(config_dict, file, indent=4)
        print(f"[Trainer] Successfully saved training argments.")
        
        # load aggregator
        if "last" not in self.config.tokens and "pooler" not in self.config.tokens:
            if "xattn" in self.config.agg_name.lower(): 
                self.aggregator = create_aggregator('xattn', config.encoder).to(self.device)
            elif 'ld2g' in self.config.agg_name.lower():
                self.aggregator = create_aggregator('ld2g', config.encoder).to(self.device)

            agg_params = sum(p.numel() for p in self.aggregator.parameters() if p.requires_grad)
            print(f"[Trainer] Build aggregator {self.config.agg_name}, trainable param {agg_params / 1e6} M")
        else:
            self.aggregator = None
        # load mil head
        mil_model_name = f"{config.mil_name}.{config.cfg_name}.{config.encoder}"
        self.mil = create_model(mil_model_name, num_classes=self.config.num_labels).to(self.device)
        mil_params = sum(p.numel() for p in self.mil.parameters() if p.requires_grad)
        print(f"[Trainer] Build mil {mil_model_name}, trainable param {mil_params / 1e6} M")

        # global batch size
        global_batch_size = self.config.batch_size * self.config.grad_acc_step
        # set gradient accumulation for CLAM and DFTD
        # if self.config.mil_name.lower() in ['clam', 'dftd'] or self.config.layer is not None:
        if self.config.mil_name.lower() in ['clam', 'dftd']:
            self.config.batch_size = 1
            self.config.grad_acc_step = global_batch_size

        print(f"[Trainer] batch size per step {self.config.batch_size}")
        print(f"[Trainer] gradient accumulation steps {self.config.grad_acc_step}")
        # initialize self.train/valid_dataloader
        self._init_dataloaders()
        # initialize self.optimization
        self._init_optimizer()
        # initialize self.best_metrics and self.metrics and confusion matrix
        self._init_metrics()
    
    def train(self):
        csv_path = os.path.join(self.ckpt_save_path, 'metrics.csv')
        headers = list(self.best_metrics.keys())
        csv_file = open(csv_path, 'w', newline='')
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        print(f"[Trainer] Metrics will be stored in \n{csv_path}")

        # train with early stop
        self.decline_epoch = 0
        for epoch in range(1, self.config.num_epochs + 1):
            train_res = self.train_epoch(epoch)
            test_res = self.evaluate(epoch, save_ckpt=True)
            print_metrics(train_res, test_res)
            # write into csv
            writer.writerow(test_res)
            csv_file.flush()         
            if self.decline_epoch == 10:
                print(f"[Main] Early stop at epoch {epoch}")
                break

        csv_file.close()
        print(f"[Trainer] Metrics saved in \n{csv_path}")
    
    def _init_dataloaders(self):
        # print("valid mode" if self.config.valid else "test mode")
        if self.config.valid:
            self.config.split = 'valid'
            self.config.train_json_path = os.path.join(self.config.file_folder, 'nfi_train.json')
            self.config.test_json_path = os.path.join(self.config.file_folder, 'nfi_valid.json')
        else:
            self.config.split = 'test'
            self.config.train_json_path = [
                os.path.join(self.config.file_folder, 'nfi_train.json'),
                os.path.join(self.config.file_folder, 'nfi_valid.json'),
            ]
            self.config.test_json_path = os.path.join(self.config.file_folder, 'nfi_test.json')
        
        # Load dataset
        train_set = NeonatalFundusDataset(self.config.train_json_path, split=self.config.split, feature_type=self.config.feature_type, layer=self.config.layer, tokens=self.config.tokens)
        print(f'[Dataset] Successfully loaded {len(train_set)} train samples.')
        test_set = NeonatalFundusDataset(self.config.test_json_path, split=self.config.split, feature_type=self.config.feature_type, layer=self.config.layer, tokens=self.config.tokens)
        print(f'[Dataset] Successfully loaded {len(test_set)} test samples.')
        
        if self.config.resample:
            print(f'[DataLoader] Randomly resample {self.config.resample_ratio} normal dataset for training.')
            self.resampler = NeonatalFundusResampler(
                dataset=train_set,
                seed=self.config.seed,
                select_ratio=self.config.resample_ratio
            )
            print(f'[DataLoader] Total training_samples {len(self.resampler)}')
        else:
            self.resampler = None
        
        # dataloader
        self.train_dataloader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            collate_fn=batch_collate_fn,
            sampler=self.resampler,
            pin_memory=True,
            num_workers=1,
            # persistent_workers=True,
            # prefetch_factor=4,
        )
        self.test_dataloader = DataLoader(
            test_set,
            batch_size=self.config.batch_size,
            collate_fn=batch_collate_fn,
            pin_memory=True,
            num_workers=1,
            # persistent_workers=True,
            # prefetch_factor=4,
        )        

    def show_trainable_params(self):
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"[Optimizer]  Pram {i + 1}:")
            print(f"[Optimizer]    lr: {group['lr']}")
            total_params = 0
            for param in group['params']:
                if param.requires_grad:
                    total_params += param.numel()
            print(f"[Optimizer]    Trainable param: {total_params:,}")

    def _init_optimizer(self):
        param_groups = [{"params": self.mil.parameters(), "lr": self.config.mil_lr, "weight_decay": self.config.weight_decay}]
        print(f"[Optimizer] MIL lr {self.config.mil_lr}")
        if self.aggregator is not None:
            param_groups.append({"params": self.aggregator.parameters(), "lr": self.config.agg_lr, "weight_decay": self.config.weight_decay})
            print(f"[Optimizer] AGG lr {self.config.agg_lr}")

        self.optimizer = torch.optim.AdamW(param_groups)
        # self.show_trainable_params()
        total_steps = self.config.num_epochs * len(self.train_dataloader)
        print(f"[Optimizer] Total training steps: {total_steps}")
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )
    
    def _init_metrics(self):
        self.best_metrics = {
            "epoch": 0,
            "AUC": 0.0, 
            "Sensitivity": 0.0, 
            "Specificity": 0.0, 
            "F1": 0.0, 
            "ACC": 0.0, 
            "Precision": 0.0
        }
        self.train_metrics = []
        self.test_metrics = []
        self.test_labelwise_metrics = []
        self.confusion_matrix = np.zeros((self.config.num_labels, self.config.num_labels))

    def train_epoch(self, epoch: int):
        # epoch start from 1
        self.mil.train()
        step_epoch = len(self.train_dataloader)
        # only main loss
        total_loss = 0
        total_bp_loss = 0
        # define pred_list and gt_list
        preds = torch.tensor([], device=self.device)
        gts = torch.tensor([], device=self.device)

        grad_acc_step = self.config.grad_acc_step
        assert grad_acc_step >= 1, "grad_acc_step must be >= 1"
        
        for i, batch in enumerate(tqdm(self.train_dataloader, total=step_epoch, desc=f'Epoch {epoch}',ncols=50) ):
            # self.optimizer.zero_grad()
            if i % grad_acc_step == 0:
                self.optimizer.zero_grad()

            image_features = batch['features'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            masks = batch['masks'].to(self.device, non_blocking=True) 
            # an image is represented by a feature map rather than a vector
            if image_features.ndim == 4:
                # todo: add aggregator
                if image_features.shape[-2] > 1:
                    if self.aggregator is not None:
                        image_features = self.aggregator(image_features, instance_mask=masks)
                    else:
                        image_features = image_features[:,:,-1,:]
                else:
                    image_features = image_features.squeeze(-2)
            cur_step = (epoch - 1) * step_epoch + i
            # print(image_features.shape)
            # print(image_features, labels.shape, masks.shape)
            outputs = self.mil(image_features, self.loss_fn, labels, masks)
            logits, loss = outputs[0]['logits'], outputs[0]['loss']
            
            total_loss += loss.item()
            loss /= self.config.grad_acc_step
            loss.backward()

            preds = torch.cat([preds, logits], dim=0)
            gts = torch.cat([gts, labels], dim=0)
            # total_bp_loss += bp_loss.item()
            # update model weights
            if (i + 1) % grad_acc_step == 0 or (i + 1) == step_epoch:
                clip_grad_norm_(self.mil.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
            # if (i + 1) % grad_acc_step == 4 or (i + 1) == step_epoch:
            #     del image_features, labels, masks, outputs, logits, loss
            #     torch.cuda.empty_cache()
            #     gc.collect()
                
        del image_features, labels, masks, outputs, logits, loss
        torch.cuda.empty_cache()
        gc.collect()

        pre, sens, spec, f1, auc, acc = specificity_precision_recall_f1_auc_acc(preds.detach().cpu().numpy(), gts.cpu().numpy())
        metrics = {
            'epoch': epoch,'AUC': auc, 'Sensitivity': sens, 
            'Specificity': spec, 'F1': f1, 'ACC': acc, 'Precision': pre
        }
        print(f"Epoch {epoch} | loss {total_loss / len(self.train_dataloader)}")
        print(f"Epoch {epoch} | loss {total_bp_loss / len(self.train_dataloader)}")
        self.train_metrics.append(metrics)
        return metrics
    
    @torch.no_grad()
    def evaluate(self, epoch, save_ckpt=False):
        self.mil.eval()
        epoch_step = len(self.test_dataloader)
        preds = torch.tensor([], device=self.device)
        gts = torch.tensor([], device=self.device)
        for i, batch in enumerate(tqdm(self.test_dataloader, desc=f'Epoch {epoch}', ncols=50)):
            
            image_features = batch['features'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            masks = batch['masks'].to(self.device, non_blocking=True) 
            
            # an image is represented by a feature map rather than a vector
            if image_features.ndim == 4:
                # todo: add aggregator
                if image_features.shape[-2] > 1:
                    if self.aggregator is not None:
                        image_features = self.aggregator(image_features, instance_mask=masks)
                    else:
                        image_features = image_features[:,:,-1,:]
                else:
                    image_features = image_features.squeeze(-2)
            outputs = self.mil(image_features, self.loss_fn, labels, masks)
            logits = outputs[0]['logits']

            preds = torch.cat([preds, logits], dim=0)
            gts = torch.cat([gts, labels], dim=0)
            
            # empty cache
        del image_features, labels, masks, outputs, logits
        torch.cuda.empty_cache()
        gc.collect()

        gts = gts.cpu().numpy()
        preds = preds.cpu().numpy()
        pre, sens, spec, f1, auc, acc = specificity_precision_recall_f1_auc_acc(preds, gts)
        metrics = {
            'epoch': epoch,'AUC': auc, 'Sensitivity': sens, 
            'Specificity': spec, 'F1': f1, 'ACC': acc, 'Precision': pre
        }

        pres, senss, specs, f1s, aucs, accs = specificity_precision_recall_f1_auc_acc(preds, gts, average_mode="none")
        labelwise_metrics = {
            'epoch': epoch,'AUC': aucs, 'Sensitivity': senss, 
            'Specificity': specs, 'F1': f1s, 'ACC': accs, 'Precision': pres
        }
        print_label_metrics(labelwise_metrics)

        self.test_metrics.append(metrics)
        self.test_labelwise_metrics.append(labelwise_metrics)

        # decide whether to save checkpoint
        score = calculate_score(metrics)
        best_score = calculate_score(self.best_metrics)
        if score < best_score:
            self.decline_epoch += 1
        else:
            self.decline_epoch = 0
        
        # score = 0.6 * metrics['AUC'] + 0.3 * metrics['Sensitivity'] + 0.1 * metrics['F1']
        # best_score = 0.6 * self.best_metrics['AUC'] + 0.3 * self.best_metrics['Sensitivity'] + 0.1 * self.best_metrics['F1']
        if score > best_score:
            print(f"New highest score {score}, history best {best_score}")
            # Todo: visualize confusion matrix
            if save_ckpt:
                self.save_checkpoint(metrics)
            # update best metrics
            self.best_metrics = metrics
            # update confusion metrics
            confusion = np.zeros((self.config.num_labels, self.config.num_labels))
            preds = np.argmax(preds, axis=1)
            for gt, pred in zip(gts, preds):
                confusion[int(gt)][int(pred)] += 1
            self.confusion = confusion
            visualize_errors(self.confusion, save_path=os.path.join(self.ckpt_save_path, f'confusion_matrix.png'))
        return metrics

    def save_checkpoint(self, metrics):
        if self.best_metrics['epoch'] > 0:
            history_best = os.path.join(self.ckpt_save_path, f"{self.best_metrics['epoch']}Epoch")
            import shutil 
            shutil.rmtree(history_best)
        # sub path in MODEL_SAVE_PATH
        save_path = os.path.join(self.config.exp_name, f"{self.config.mil_name}-{self.config.exp_idx}", f"{metrics['epoch']}Epoch")
        print("[Trainer] Save best performed checkpoint.")
        save_model(self.mil, save_path, save_pretrained=True)
