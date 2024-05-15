import os

os.environ["WANDB_DIR"] = os.path.abspath("wandb")
os.environ["TMPDIR"] = os.path.abspath("wandb")

import warnings

warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from scene_graph_prediction.scene_graph_helpers.dataset.adverserial_or_dataset import AdverserialORDataset
from scene_graph_prediction.scene_graph_helpers.model.adverserial_wrapper_oracle import AdverserialOracleWrapper


def config_loader(config_path: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers/configs') / config_path
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config


def load_checkpoint_data(file_path):
    if Path(file_path).exists():
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}


def update_checkpoint_data(file_path, model_name, checkpoint_id, wandb_run_id=None):
    data = load_checkpoint_data(file_path)
    if model_name not in data:
        data[model_name] = {"checkpoints": [], "wandb_run_id": wandb_run_id}
    if checkpoint_id not in data[model_name]["checkpoints"]:
        data[model_name]["checkpoints"].append(checkpoint_id)
    if wandb_run_id:
        data[model_name]["wandb_run_id"] = wandb_run_id
    with open(file_path, 'w') as file:
        json.dump(data, file)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    config = config_loader(args.config)

    print(f'Model path: {args.model_path}')
    eval_dataset = AdverserialORDataset(config)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)  # config['NUM_WORKERS']
    model = AdverserialOracleWrapper(config, labels=eval_dataset.labels, model_path=args.model_path)
    model.validate(eval_loader)


if __name__ == '__main__':
    import subprocess

    subprocess.call(['nvidia-smi', '-L'])
    main()
