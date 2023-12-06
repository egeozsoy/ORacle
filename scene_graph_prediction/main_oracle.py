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
    mode = 'evaluate'  # can be evaluate/infer

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
    # TODO repeat log vs linear test
    # TODO see if 50 epochs are really necessary. Maybe 20 is enough.
    # TODO only changes. OR/OP scene graph. Used as memory.

    #             28120    part-1   oracle ege_oezs  R      17:07      1 unimatrix1 # temporal training
    #             28132    part-1   oracle ege_oezs  R       0:02      1 unimatrix1 # 20 perm, 12 unfreeze, nontemporal training

    # TODO eval 10_perm and 20_perm as well. but 100 perm looks good!
    # TODO temporality using surgery SG. needs adaptation in train and eval. Initial version uses GT, should be online at some point
    # TODO novel view evaluvation.
    # TODo for multiview: Nximages -> Clip Model -> NxCLIP embeddings (Nx576x1024) -> ImagePooler(2-4 layers max) -> CLIP Embedding(576x1024) -> Projection layer -> (576x4096) -> LLM
    # Cleanup helper: rm -rf */checkpoint-*/global_step*z

    import subprocess

    subprocess.call(['nvidia-smi', '-L'])
    main()
