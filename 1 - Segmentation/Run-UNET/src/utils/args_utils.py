import argparse
import yaml

# specifying the training params
def train_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp_id',type=str, default='exp/0')
    parser.add_argument('--wandb', action='store_true', help='If true run wandb logger')
    parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--lr', type=float, default='1e-4') #1e-4 is best
    parser.add_argument('--bs', type=int, default=16) # change if there's memory issues
    parser.add_argument('--epoch', type=int, default=30)
    args = parser.parse_args()
    return args

def test_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir',type=str, default='/home/syurtseven/gsoc/scripts/results')
    parser.add_argument('--seed',type=int, default=31)
    parser.add_argument('--sample_size', type=int, default=30)
    parser.add_argument('--model_path', type=str, default='')
    args = parser.parse_args()
    return args