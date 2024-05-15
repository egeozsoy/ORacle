import random
import warnings
from collections import Counter
from random import shuffle
import transformers

from scene_graph_prediction.llava_helpers.generate_dataset_format_for_llava import generate_finetuning_samples_from_dataset
from scene_graph_prediction.llava_helpers.scene_graph_converters import extract_take_int_from_image_path, parse_llava_sg, llava_sg_to_surgery_sg, surgery_sg_to_memory_str
from synthetic_or_generation.generate_synthetic_dataset_format_for_llava import generate_finetuning_samples

warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl

from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset


def config_loader(config_path: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers/configs') / config_path
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config


def main():
    N_PERM = 10  # TODO this leads to a 50% 50% split between real and fake samples.
    ADD_TEMPORAL = False
    WITH_TEMPORAL_AUG = False
    MEMORY_INDICATOR = 'double'  # single: Memory, double: <memory_start> and <memory_end>
    TEMPORAL_STYLE = 'longshort'  # can be longshort or all or longshort_compact
    INCLUDE_TIMEPOINT = False
    DROP_HISTORY = 0.5  # either False or float
    SG_INDICATOR = 'double'  # double: <SG> and </SG>
    SPLIT = 'train'
    # views_to_use = (2,)
    views_to_use = (2, 1, 3, 5)
    FAKE_ATTRIBUTES = True
    FAKE_P = 0.5
    COT_PROMPTING = True  # chain of thought prompting
    WITHOUT = []  # ['drilling', 'hammering', 'sawing']
    # TODO FLAG FOR TIMEPOINT and DROPPING. Naming Scheme should be so that only interesting flags are including in the name, not if they are False.
    NAME = f'{SPLIT}_{N_PERM}perm_{ADD_TEMPORAL}temp_{MEMORY_INDICATOR}mem_{WITH_TEMPORAL_AUG}tempaug_{TEMPORAL_STYLE}_symbolic_{SG_INDICATOR}sg_merged'
    if FAKE_ATTRIBUTES:
        NAME += f'_fake_attributes_{FAKE_P}'
    if not INCLUDE_TIMEPOINT:
        NAME += '_notimepoints'
    if DROP_HISTORY is not False and DROP_HISTORY > 0.01:
        NAME += f'_drophistory{DROP_HISTORY}'
    if COT_PROMPTING:
        NAME += '_cot'
    if len(views_to_use) > 1:
        NAME += f'_{len(views_to_use)}views'
    if len(WITHOUT) > 0:
        NAME += f'_without{"_".join(WITHOUT)}'
    print(f'Creating samples for LLAVA dataset with name {NAME}')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    config = config_loader(args.config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'liuhaotian/llava-v1.5-7b',
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )

    dataset = ORDataset(config, SPLIT)

    real_samples = generate_finetuning_samples_from_dataset(dataset, n_permutations=N_PERM, SG_INDICATOR=SG_INDICATOR, INCLUDE_TIMEPOINT=INCLUDE_TIMEPOINT, SYMBOLIC_SG=True,
                                                            views_to_use=views_to_use)

    if len(views_to_use) > 1:
        if len(WITHOUT) > 0:
            dataset_path = Path(f'/home/guests/shared/Oracle/synthetic_4D-OR_mv_without{"_".join(WITHOUT)}')
        else:
            dataset_path = Path('/home/guests/shared/Oracle/synthetic_4D-OR_mv')
    else:
        if len(WITHOUT) > 0:
            dataset_path = Path(f'synthetic_or_generation/synthetic_4D-OR_without{"_".join(WITHOUT)}')
        else:
            dataset_path = Path('synthetic_or_generation/synthetic_4D-OR')

    fake_samples = generate_finetuning_samples(dataset_path, views_to_use=views_to_use,
                                               SG_INDICATOR=SG_INDICATOR, INCLUDE_TIMEPOINT=INCLUDE_TIMEPOINT,
                                               SYMBOLIC_SG=True, FAKE_ATTRIBUTES=FAKE_ATTRIBUTES, FAKE_P=FAKE_P, COT_PROMPTING=COT_PROMPTING)

    # Load the tokenizer which will be used
    # val_samples = generate_finetuning_samples_from_dataset(val_dataset)
    # Also calculate the corresponding word frequencies

    if ADD_TEMPORAL:
        print('Adding temporal information...')
        take_to_history = {}
        take_timepoint_to_memory_str = {}

        for take_int in range(1, 11):
            take_scene_graphs = [elem for elem in real_samples if extract_take_int_from_image_path(elem['image']) == take_int]
            # make unique
            take_scene_graphs = list({elem['timepoint']: elem for elem in take_scene_graphs}.values())
            # sort by timepoint
            take_scene_graphs = sorted(take_scene_graphs, key=lambda x: x['timepoint'])
            take_scene_graphs_reformatted = []
            for take_scene_graph in take_scene_graphs:
                scene_graph = parse_llava_sg(take_scene_graph['conversations'][1]['value'])
                take_scene_graphs_reformatted.append({'timepoint_idx': take_scene_graph['timepoint'], 'scene_graph': scene_graph})
            surgery_sg_triplets = llava_sg_to_surgery_sg(take_scene_graphs_reformatted, entity_of_interest=None, IRRELEVANT_PREDS=['closeto', 'closeTo'])
            with open(f'data/llava_samples/surgery_sg_{take_int}.json', 'w') as f:
                json.dump(surgery_sg_triplets, f)
            take_to_history[take_int] = surgery_sg_triplets

        llava_scene_graphs_with_history = []
        for llava_scene_graph in real_samples:
            image_path = llava_scene_graph['image'] if isinstance(llava_scene_graph['image'], str) else llava_scene_graph['image'][0]
            image_path = Path(image_path)
            take_int = extract_take_int_from_image_path(image_path)
            surgery_sg_triplets = take_to_history[take_int]
            timepoint = llava_scene_graph['timepoint']
            surgery_sg_triplets = [elem for elem in surgery_sg_triplets if elem[0] < timepoint]
            memory_str = surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint=timepoint, TEMPORAL_STYLE=TEMPORAL_STYLE, INCLUDE_TIMEPOINTS=INCLUDE_TIMEPOINT)
            take_timepoint_to_memory_str[f'{take_int}_{timepoint}'] = memory_str
            input = llava_scene_graph['conversations'][0]['value']

            if WITH_TEMPORAL_AUG:
                p = random.random()
                if p < 0.5:
                    memory_str = None
                elif p < 0.666:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint=timepoint, TEMPORAL_STYLE='short', INCLUDE_TIMEPOINTS=INCLUDE_TIMEPOINT, DROP_HISTORY=DROP_HISTORY)
                elif p < 0.833:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint=timepoint, TEMPORAL_STYLE='long', INCLUDE_TIMEPOINTS=INCLUDE_TIMEPOINT, DROP_HISTORY=DROP_HISTORY)
                else:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint=timepoint, TEMPORAL_STYLE='longshort', INCLUDE_TIMEPOINTS=INCLUDE_TIMEPOINT, DROP_HISTORY=DROP_HISTORY)

            if memory_str is not None:
                if MEMORY_INDICATOR == 'single':
                    input = input.replace('<image>\n', f'<image>\nMemory: {memory_str}.')
                elif MEMORY_INDICATOR == 'double':
                    input = input.replace('<image>\n', f'<image>\n<memory_start>: {memory_str}<memory_end>.')
                else:
                    raise NotImplementedError

            # input = input.replace('Describe this image', 'Describe this image at timepoint T')  # add timepoint to the question
            llava_scene_graph['conversations'][0]['value'] = input
            llava_scene_graphs_with_history.append(llava_scene_graph)

        real_samples = llava_scene_graphs_with_history

        with open(f'data/llava_samples/{NAME}_take_timepoint_to_memory_str.json', 'w') as f:
            json.dump(take_timepoint_to_memory_str, f)

    samples = real_samples + fake_samples
    token_freq = Counter()
    longest_sample = -1
    for sample in samples:
        for conversation in sample['conversations']:
            if conversation['from'] == 'gpt':
                tokenized = tokenizer.tokenize(conversation['value'])
                token_freq.update(tokenized)
                longest_sample = max(longest_sample, len(tokenized))

    # randomly shuffle the samples
    shuffle(samples)

    with open(f'data/llava_samples/{NAME}.json', 'w') as f:
        json.dump(samples, f, indent=4)

    if SPLIT == 'train' and not ADD_TEMPORAL:
        if len(WITHOUT) > 0:
            with open(f'data/llava_samples/train_token_freqs_7b_symbolic_merged_removal_{FAKE_P}_{COT_PROMPTING}_without{"_".join(WITHOUT)}.json', 'w') as f:
                json.dump(token_freq, f, indent=4)
        else:
            with open(f'data/llava_samples/train_token_freqs_7b_symbolic_merged_removal_{FAKE_P}_{COT_PROMPTING}.json', 'w') as f:
                json.dump(token_freq, f, indent=4)


if __name__ == '__main__':
    """format of json (ultimately):
    1) id
    2) image(s) paths
    3) Prompt (formulately in multiple ways)
    4) Answer (formulately in multiple ways) (multiple orders)
    5) Prompt should include knowledge about different things
    
    Optionally: include augmentations, modifications in scene graph, prompt or both etc.
    """
    import subprocess

    subprocess.call(['nvidia-smi', '-L'])
    main()
