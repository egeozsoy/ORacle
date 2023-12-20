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

from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset
from scene_graph_prediction.scene_graph_helpers.model.scene_graph_prediction_model_oracle import OracleWrapper


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
    mode = 'eval_all'  # can be evaluate/infer/eval_all

    name = args.config.replace('.json', '')

    if mode == 'evaluate':
        print(f'Model path: {args.model_path}')
        train_dataset = ORDataset(config, 'train', shuffle_objs=True)
        eval_dataset = ORDataset(config, 'val')
        # eval_dataset = ORDataset(config, 'train')
        eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                 collate_fn=eval_dataset.collate_fn)
        model = OracleWrapper(config, num_class=len(eval_dataset.classNames), num_rel=len(eval_dataset.relationNames),
                              weights_obj=train_dataset.w_cls_obj,
                              weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames,
                              model_path=args.model_path)
        model.validate(eval_loader)
    elif mode == 'eval_all':
        print('Evaluating all checkpoints')

        evaluated_file = 'evaluated_checkpoints.json'
        checkpoint_data = load_checkpoint_data(evaluated_file)
        model_path = Path(args.model_path)
        model_name = model_path.name
        eval_every_n_checkpoints = 4
        wandb_run_id = checkpoint_data.get(model_name, {}).get("wandb_run_id", None)
        logger = pl.loggers.WandbLogger(project='oracle_evals', name=model_name, save_dir='logs', offline=False, id=wandb_run_id)
        train_dataset = ORDataset(config, 'train', shuffle_objs=True)
        eval_dataset = ORDataset(config, 'val')
        eval_dataset_for_train = ORDataset(config, 'train')
        # always eval last checkpoint
        checkpoints = sorted(list(model_path.glob('checkpoint-*')), key=lambda x: int(str(x).split('-')[-1]))
        print(checkpoints)
        checkpoint_idx = 0
        while checkpoint_idx < len(checkpoints):
            checkpoint = checkpoints[checkpoint_idx]
            if checkpoint_idx % eval_every_n_checkpoints != 0 and checkpoint_idx != len(checkpoints) - 1:
                print(f'Skipping checkpoint: {checkpoint}')
                checkpoint_idx += 1
                continue
            if checkpoint_idx == 0 and 'continue' not in model_name:
                print(f'Skipping checkpoint: {checkpoint}')
                continue
            checkpoint_id = int(checkpoint.name.split('-')[-1])
            if model_name in checkpoint_data and checkpoint_id in checkpoint_data[model_name]["checkpoints"]:
                print(f'Checkpoint {checkpoint_id} for model {model_name} already evaluated. Skipping.')
                checkpoint_idx += 1
                continue
            print(f'Evaluating checkpoint: {checkpoint}...')
            train_loader = DataLoader(eval_dataset_for_train, batch_size=8, shuffle=True, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                      collate_fn=eval_dataset.collate_fn)
            eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                     collate_fn=eval_dataset.collate_fn)
            model = OracleWrapper(config, num_class=len(eval_dataset.classNames), num_rel=len(eval_dataset.relationNames),
                                  weights_obj=train_dataset.w_cls_obj,
                                  weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames,
                                  model_path=str(checkpoint))
            model.validate(train_loader, limit_val_batches=125, logging_information={'split': 'train', "logger": logger, "checkpoint_id": checkpoint_id})
            model.validate(eval_loader, logging_information={'split': 'val', "logger": logger, "checkpoint_id": checkpoint_id})
            update_checkpoint_data(evaluated_file, model_name, checkpoint_id, logger.experiment.id)
            checkpoint_idx += 1
            checkpoints = sorted(list(model_path.glob('checkpoint-*')), key=lambda x: int(str(x).split('-')[-1]))  # update checkpoints in case new ones were added

    elif mode == 'infer':
        raise NotImplementedError('TODO')
        train_dataset = ORDataset(config, 'train', shuffle_objs=True)
        infer_split = 'test'
        eval_dataset = ORDataset(config, infer_split, for_eval=True)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                 collate_fn=eval_dataset.collate_fn)

        if paper_weight is not None:
            model = OracleWrapper(config, num_class=len(eval_dataset.classNames), num_rel=len(eval_dataset.relationNames),
                                  weights_obj=train_dataset.w_cls_obj,
                                  weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)
            model.load_state_dict(torch.load(paper_weight))
            checkpoint_path = None
        else:
            # checkpoint_path = None # Can hardcode a difference checkpoint path here if wanted
            model = OracleWrapper.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, num_class=len(eval_dataset.classNames),
                                                       num_rel=len(eval_dataset.relationNames), weights_obj=train_dataset.w_cls_obj,
                                                       weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)
        trainer = pl.Trainer(gpus=1, max_epochs=config['MAX_EPOCHES'], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=50, num_sanity_val_steps=0,
                             callbacks=[pl.callbacks.progress.RichProgressBar()])
        assert checkpoint_path is not None or paper_weight is not None
        results = trainer.predict(model, eval_loader, ckpt_path=checkpoint_path)
        scan_relations = {key: value for key, value in results}
        output_name = f'scan_relations_{name}_{infer_split}.json'
        with open(output_name, 'w') as f:
            json.dump(scan_relations, f)
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    # TODO augmentation + 50 epochs
    # TODO don't init pooler if not used.
    # TODO decide grad accum or not

    # TODO only train with one take or a few
    # TODO temporality using surgery SG. needs adaptation in train and eval. Initial version uses GT, should be online at some point+
    # TODO novel view evaluvation.
    # TODO novel action/role evaluation.
    # TODO wrong object assisgnments with description. Task assignments wrong.
    # TODO phase description based on scene graphs -> predict scene graph.
    # TODO abnormality detection. Knowledge describes what is normal.
    # TODO red circle based augmentations
    # TODO sterility breech detection
    # TODO object1 object2 etc and what they mean are hidden in the knowledge. The knowledge can be as explicit as object1: head_surgeon object2: drill but also something like object1: -is a person that does X,Y,Z. Analogous for relations.
    # TODO fake patient knowledge: Aneshtesia or not.
    # TODo for multiview: Nximages -> Clip Model -> NxCLIP embeddings (Nx576x1024) -> ImagePooler(2-4 layers max) -> CLIP Embedding(576x1024) -> Projection layer -> (576x4096) -> LLM
    # Cleanup helper: rm -rf */checkpoint-*/global_step*

    import subprocess

    subprocess.call(['nvidia-smi', '-L'])
    main()
