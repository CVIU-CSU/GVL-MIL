import argparse
import sys 
sys.path.insert(0, "/root/userfolder/MIL/VL-MIL")
from mil.train.mil_trainer import MILTrainer
from mil.train.utils import setup_seed


seed_list = [42, 123, 2025, 12138, 114514]

def parse_args():
    parser = argparse.ArgumentParser(description='Set training config.')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp-name", type=str, default='main')
    parser.add_argument("--exp-idx", type=int, default=0)
    # valid or test, resample or not
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--valid', action='store_true', help='Enable validation (default)')
    group.add_argument('--test', action='store_false', dest='valid', help='Disable validation')
    parser.set_defaults(valid=True) # default validation
    
    # mil args
    parser.add_argument("--mil-name", type=str)
    parser.add_argument("--agg-name", type=str, default="xattn")
    parser.add_argument("--cfg_name", type=str, default="base", help="mil configuration")
    parser.add_argument("--num-labels", type=str, default=4)
    # feature args
    parser.add_argument("--encoder", type=str, default="siglip", help='qwen2-7b, qwen2-0.5b, siglip')
    parser.add_argument("--layer", type=int, default=None, help="3,11,23,27,28 for qwen feature, others are not considered")
    parser.add_argument("--tokens", type=str, default='pooler', help="select tokens: all, image, instruct, output for qwen; pooler and all for siglip")
    # data args
    parser.add_argument("--file-folder", type=str, default="/root/userfolder/data-ckpts/VL-MIL/datasets/v2/screenings", help="for nfi_train/valid/test.json")
    # training args
    parser.add_argument("--resample-ratio", type=float, default=1.0, help="resample ratio, 1 for not resample")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size a step")
    parser.add_argument("--grad-acc-step", type=int, default=1, help="gradient accumulation steps")
    # parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--mil-lr", type=float, default=2e-5, help="learning rate of mil heads")
    parser.add_argument("--agg-lr", type=float, default=2e-5, help="learning rate of aggregatioon")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--warmup-ratio", type=int, default=0.05)
    parser.add_argument("--weight-decay", type=int, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--loss", type=str, default="CBCE", help="Loss type: CE, Focal, NLL, CBCE")
    parser.add_argument("--patience", type=int, default=10, help="For early stop")
   
    
    # parse args
    args = parser.parse_args()
    if 'qwen' in args.encoder:
        args.feature_type = 'qwen'
    elif 'siglip' in args.encoder:
        args.feature_type = 'siglip'
    else:
        raise KeyError("Unknown encoder type")
    if args.resample_ratio == 1:
        args.resample = False 
    else: 
        args.resample = True
    
    return args


# --- Optional: Example usage ---
if __name__ == "__main__":
    config = parse_args()
    print(config)
    seed = seed_list[int(config.exp_idx) % len(seed_list)]
    setup_seed(seed)
    config.seed = seed

    trainer = MILTrainer(config)
    print("[Main] Successfully loaded trainer")
    print(trainer.aggregator)
    trainer.train()
    # main(config)    
