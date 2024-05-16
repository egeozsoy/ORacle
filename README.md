# ORacle

Official code of the paper ORacle: Large Vision-Language Models for Knowledge-Guided Holistic OR Domain Modeling accepted at MICCAI 2024.

## What is not included in this repository?

The 4D-OR Dataset itself, is not included in this project, please refer to [4D-OR](https://github.com/egeozsoy/4D-OR) repository for information on downloading the dataset. The rest of this README
assumes the 4D-OR repository is located in the same folder as the root project. You can modify this in: helpers/configurations.py OR_4D_DATA_ROOT_PATH

TODO adversarial_4dor should not be part of repo but made available. (on huggingface)

## Installation

- Run `pip install -r requirements.txt`.
- cd instal LLaVA then run `pip install -e .` to install the correct version of the LLaVA library.
- Potentially you need to explicitly install flash-attn like `pip install flash-attn --no-build-isolation`

## Data Generation for LVLM Training

- If you want to train on the original 4D-OR dataset, follow this. If you want to use the synthetic dataset, refer to that section
- To generate the training json run `python -m scene_graph_prediction.llava_helpers.generate_dataset_format_for_llava`
- Set the ADD_TEMPORAL to True for generating data for temporal training, or set the Symbolic SG to True for generating data for symbolic scene graph training.

## Synthetic Data Generation

- If you want to generate and train a synthetic dataset, follow this.
- First download entity_crops_all from https://huggingface.co/egeozsoy/ORacle/resolve/main/entity_crops_all.zip?download=true, unzip and place it into synthetic_or_generation/entity_crops_all
- Run `python -m synthetic_or_generation.generate_novel_entities` to generate many tools for the synthetic dataset. You need to first download the DIS model
  from https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth?download=true and put in into the path specified in the script as dis_model_path. This step can be
  run as much as you desire, we suggest at least 100000.
- After the entities are generated, it is advisable to prune the ones that do not look meaningful, to this end use `python -m synthetic_or_generation.prune_novel_entities`
- Now you can generate the synthetic dataset by running `python -m synthetic_or_generation.generate_novel_augmentations`
- TODO generate jsons
- TODO visual descriptor data generation

## Training

- TODO how to train normal model, temporal model, symbolic model
- We use SLURM for all our training. If you have a setup that does not use SLURM, you can still use the configs in our slurm scripts. Usually, simply running the part after "srun" would work as well.
- Our base model is our multiview model, without any temporality. To train this, use the slurm config (from inside the LLaVa Folder): `slurm_config_multiview.conf` (you will need to adapt the data
  path to your previously experted training json path). Furthermore, you need to modify LLaVA.llava.train.llava_trainer.py at the top, to correctly load the token_frequencies.
- To train our temporal model, we prefer to use curriculum learning, starting from the base model. To this, you have to already do the previous step. Afterwards, you can
  run `slurm_config_multiview_temporal_curriculum.conf`. (you will again need to adapt the paths)
- To train our symbolic model, . Make sure to set token frequencies to None in LLaVA.llava.train.llava_trainer.py.
- To train our symbolic model with visual prompts,
- TODO How to input textual or visual prompts

## Evaluation

- Instead of training models from scratch, you can download the pretrained models from the following links:
- TODO normal eval
- TODO adversarial eval
- TODO description of how to use textual or visual prompts here