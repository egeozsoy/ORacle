# ORacle: Large Vision-Language Models for Knowledge-Guided Holistic OR Domain Modeling

<img align="right" src="figures/teaser.jpg" alt="teaser" width="100%" style="margin-left: 10px">

Official code of the paper ORacle: Large Vision-Language Models for Knowledge-Guided Holistic OR Domain Modeling accepted at MICCAI 2024 (https://arxiv.org/abs/2404.07031). Every day, countless surgeries are performed worldwide, each within the distinct settings of operating rooms (ORs) that vary not only in their setups but also in the personnel, tools, and equipment used. 
This inherent diversity poses a substantial challenge for achieving a holistic understanding of the OR, as it requires models to generalize beyond their initial training datasets. 
To reduce this gap, we introduce ORacle, an advanced vision-language model designed for holistic OR domain modeling, which incorporates multi-view and temporal capabilities and can leverage external knowledge during inference, enabling it to adapt to previously unseen surgical scenarios. 
This capability is further enhanced by our novel data augmentation framework, which significantly diversifies the training dataset, ensuring ORacle's proficiency in applying the provided knowledge effectively. 
In rigorous testing, in scene graph generation, and downstream tasks on the 4D-OR dataset, ORacle not only demonstrates state-of-the-art performance but does so requiring less data than existing models. 
Furthermore, its adaptability is displayed through its ability to interpret unseen views, actions, and appearances of tools and equipment. 
This demonstrates ORacle's potential to significantly enhance the scalability and affordability of OR domain modeling and opens a pathway for future advancements in surgical data science.


PDF Paper: https://link.springer.com/content/pdf/10.1007/978-3-031-72089-5_43.pdf?pdf=inline%20link  
PDF Book: https://link.springer.com/chapter/10.1007/978-3-031-72089-5_43

**Authors**: [Ege Özsoy][eo]\*, [Chantal Pellegrini][cp]\*, [Matthias Keicher][mk], [Nassir Navab][nassir]

[eo]:https://www.cs.cit.tum.de/camp/members/ege-oezsoy/

[cp]:https://www.cs.cit.tum.de/camp/members/chantal-pellegrini/

[mk]:https://www.cs.cit.tum.de/camp/members/matthias-keicher/

[nassir]:https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/

```
@inproceedings{ozsoy2024oracle,
  title={ORacle: Large Vision-Language Models for Knowledge-Guided Holistic OR Domain Modeling},
  author={Ege Özsoy and Pellegrini, Chantal and Keicher, Matthias and Navab, Nassir},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={455--465},
  year={2024},
  organization={Springer}
}

@inproceedings{Özsoy2023_LABRAD_OR,
    title={LABRAD-OR: Lightweight Memory Scene Graphs for Accurate Bimodal Reasoning in Dynamic Operating Rooms},
    author={Ege Özsoy, Tobias Czempiel, Felix Holm, Chantal Pellegrini, Nassir Navab},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    year={2023},
    organization={Springer}
}
@Article{Özsoy2023,
author={{\"O}zsoy, Ege
and Czempiel, Tobias
and {\"O}rnek, Evin P{\i}nar
and Eck, Ulrich
and Tombari, Federico
and Navab, Nassir},
title={Holistic OR domain modeling: a semantic scene graph approach},
journal={International Journal of Computer Assisted Radiology and Surgery},
year={2023},
doi={10.1007/s11548-023-03022-w},
url={https://doi.org/10.1007/s11548-023-03022-w}
}
@inproceedings{Özsoy2022_4D_OR,
    title={4D-OR: Semantic Scene Graphs for OR Domain Modeling},
    author={Ege Özsoy, Evin Pınar Örnek, Ulrich Eck, Tobias Czempiel, Federico Tombari, Nassir Navab},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    year={2022},
    organization={Springer}
}
@inproceedings{Özsoy2021_MSSG,
    title={Multimodal Semantic Scene Graphs for Holistic Modeling of Surgical Procedures},
    author={Ege Özsoy, Evin Pınar Örnek, Ulrich Eck, Federico Tombari, Nassir Navab},
    booktitle={Arxiv},
    year={2021}
}
```

## What is not included in this repository?

The 4D-OR Dataset itself, is not included in this project, please refer to [4D-OR](https://github.com/egeozsoy/4D-OR) repository for information on downloading the dataset. The rest of this README
assumes the 4D-OR repository is located in the same folder as the root project. You can modify this in: helpers/configurations.py OR_4D_DATA_ROOT_PATH

## Installation

- Run `pip install -r requirements.txt`.
- cd instal LLaVA then run `pip install -e .` to install the correct version of the LLaVA library.
- Potentially you need to explicitly install flash-attn like `pip install flash-attn --no-build-isolation`
- If you need to download any file, data, model weight etc., they are likely located at https://huggingface.co/egeozsoy/ORacle/tree/main

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
- Finally, you can run `python -m synthetic_or_generation.generate_synthetic_dataset_format_for_llava` to generate the jsons for training.
- If you want to use visual descriptors instead of textual descriptors, you first need to generate those by running `python -m synthetic_or_generation.generate_vis_descriptors`
- Next run `python -m synthetic_or_generation.embed_visual_prompts` to convert the visual prompts into pytorch tensors and augment them at the same time. Make sure to download and copy
  original_crops.zip to the correct folder and unzip it.
- Now run `python -m synthetic_or_generation.generate_synthetic_vis_descriptor_dataset_format_for_llava` to generate the jsons for training with visual descriptors.

## Training

- We use SLURM for all our training. If you have a setup that does not use SLURM, you can still use the configs in our slurm scripts. Usually, simply running the part after "srun" would work as well.
- Our base model is our multiview model, without any temporality. To train this, use the slurm config (from inside the LLaVa Folder): `slurm_config_multiview.conf` (you will need to adapt the data
  path to your previously experted training json path). Furthermore, you need to modify LLaVA.llava.train.llava_trainer.py at the top, to correctly load the token_frequencies.
- To train our temporal model, we prefer to use curriculum learning, starting from the base model. To this, you have to already do the previous step. Afterwards, you can
  run `slurm_config_multiview_temporal_curriculum.conf`. (you will again need to adapt the paths)
- To train our symbolic model, run `slurm_config_multiview_symbolic`. Make sure to set token frequencies to None in LLaVA.llava.train.llava_trainer.py.
- To train our symbolic model with visual prompts, run `slurm_config_multiview_symbolic_visual`. Make sure to set token frequencies to None in LLaVA.llava.train.llava_trainer.py.

## Evaluation

- Instead of training models from scratch, you can download the pretrained models from https://huggingface.co/egeozsoy/ORacle/tree/main and unzip and put into the folder LLAVA/checkpoints
- To evaluate the base model
  run `python -u -m scene_graph_prediction.main --config oracle_mv_learned.json --model_path LLaVA/checkpoints/llava-v1.5-7b-task-lora_4dor_qlora_100perm_4_view_2135_orderaug_image`
- To evaluate the temporal model
  run `python -u -m scene_graph_prediction.main --config oracle_mv_learned_temporal_pred.json --model_path LLaVA/checkpoints/llava-v1.5-7b-task-lora_4dor_qlora_100perm_4_view_2135_orderaug_image_temporal_curriculum`
- To evaluate the symbolic model
  run `python -u -m scene_graph_prediction.main --config oracle_mv_learned_symbolic.json --model_path LLaVA/checkpoints/llava-v1.5-7b-task-lora_4dor_qlora_100perm_4_view_2135_orderaug_image_synthetic`
- To evaluate the symbolic visual model
  run `python -u -m scene_graph_prediction.main --config oracle_mv_learned_symbolic_visual.json --model_path LLaVA/checkpoints/llava-v1.5-7b-task-lora_4dor_qlora_100perm_4_view_2135_orderaug_image_synthetic_visual`

#### Adaptability Benchmark

- First download adaptability_4dor.zip from huggingface (https://huggingface.co/egeozsoy/ORacle/tree/main), then unzip it into the root folder of the project.
- To evaluate the symbolic model on the adaptability benchmark
  run `python -u -m scene_graph_prediction.adaptability_4dor_eval --config oracle_mv_learned_symbolic.json --model_path LLaVA/checkpoints/llava-v1.5-7b-task-lora_4dor_qlora_100perm_4_view_2135_orderaug_image_synthetic`
- To evaluate the symbolic visual model on the adaptability benchmark
  run `python -u -m scene_graph_prediction.adaptability_4dor_eval --config oracle_mv_learned_symbolic_visual.json --model_path LLaVA/checkpoints/llava-v1.5-7b-task-lora_4dor_qlora_100perm_4_view_2135_orderaug_image_synthetic_visual`
- Finally: before uploading final results to the online eval tool, use scene_graph_prediction/map_raw_prediction_to_4dorevaluator.py to convert the predictions to the correct format. 
