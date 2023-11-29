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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    config = config_loader(args.config)
    mode = 'evaluate'  # can be train/evaluate/infer

    name = args.config.replace('.json', '')

    if mode == 'evaluate':
        print(f'Model path: {args.model_path}')
        train_dataset = ORDataset(config, 'train', shuffle_objs=True)
        eval_dataset = ORDataset(config, 'val')
        # eval_dataset = ORDataset(config, 'train')
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                 collate_fn=eval_dataset.collate_fn)
        model = OracleWrapper(config, num_class=len(eval_dataset.classNames), num_rel=len(eval_dataset.relationNames),
                              weights_obj=train_dataset.w_cls_obj,
                              weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames,
                              model_path=args.model_path)
        model.validate(eval_loader)
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
    # TODO use multiview. test concat, vs average vs max.
    # TODO How to avoid closeTo. Maybe curate the training dataset, or simply do without closeTo. Or maybe adjust the order that all none close to stuff is done first
    # TODO only evaluvate fully finished models, otherwise it is just wrong
    # TODO repeat log vs linear test
    # TODO experiment with adjusting the vocabulary. It is not trivial but should be possible
    # TODO validation during training
    # TODO we could also do a projection layer only training first to align and reduce overfitting?
    # TODO unfreeze some backbone layers

    import subprocess

    subprocess.call(['nvidia-smi', '-L'])
    main()
