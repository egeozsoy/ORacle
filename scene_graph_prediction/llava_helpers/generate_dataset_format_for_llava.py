import warnings
from collections import Counter
from random import shuffle
import transformers

from helpers.configurations import OR_4D_DATA_ROOT_PATH

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


def load_image_paths(scan_id_no_split):
    take_idx, pcd_idx = scan_id_no_split.split('_')
    with open(f'{OR_4D_DATA_ROOT_PATH}/export_holistic_take{take_idx}_processed/timestamp_to_pcd_and_frames_list.json') as f:
        timestamp_to_pcd_and_frames_list = json.load(f)
    image_paths = []
    for c_idx in range(1, 7):
        color_idx_str = timestamp_to_pcd_and_frames_list[int(pcd_idx)][1][f'color_{c_idx}']
        color_path = Path(f'{OR_4D_DATA_ROOT_PATH}/export_holistic_take{take_idx}_processed/colorimage/camera0{c_idx}_colorimage-{color_idx_str}.jpg')
        image_paths.append(color_path)

    return image_paths


def scene_graph_to_string(scene_graph, human_idx_to_name):
    '''
    Scene graph is a list of relations in the form of (subject, relation, object)
    '''
    out = ''
    for (subject, relation, object) in scene_graph:
        # optionally modify subject, relation, object here
        if 'human_' in subject:
            subject = human_idx_to_name.get(subject, 'circulator')  # default to circulator
        if 'human_' in object:
            object = human_idx_to_name.get(object, 'circulator')  # default to circulator

        subject = subject.replace('_', ' ').lower()
        object = object.replace('_', ' ').lower()
        # Convert first letter of predicate to lower case
        relation = relation[0].lower() + relation[1:]
        if relation == 'operating':
            relation = 'manipulating'

        # if subject != 'head surgeon' or object != 'patient': # TODO remove this. This only uses the main action
        #     continue
        out += f'{subject},{object},{relation}; '

    # Remove last comma, put end token
    if out:
        out = out[:-2] + '.'
    else:
        out = '.'  # empty scene graph
    return out


def apply_template(image_paths, scene_graph):
    human_prompt = 'Describe this image using a scene graph, represented as a list of triplets. Each triplet consists of a subject(entity), an object(entity), and a predicate. Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching].'
    id = f'{image_paths[0].parent.parent.stem}/{image_paths[0].stem}'

    sample = {'id': id, 'image': [str(image_path.absolute()) for image_path in image_paths] if len(image_paths) > 1 else str(image_paths[0].absolute()),
              "conversations": [
                  {
                      "from": "human",
                      "value": f"<image>\n{human_prompt}"
                  },
                  {
                      "from": "gpt",
                      "value": scene_graph
                  },
              ]
              }

    return sample


def generate_finetuning_samples_from_dataset(dataset, n_permutations=1, views_to_use=(1,2,3,5)):
    samples = []
    for index in range(len(dataset)):
        scan_id = dataset.scans[index]
        scan_id_no_split = scan_id.rsplit('_', 1)[0]
        take_idx, pcd_idx = scan_id_no_split.split('_')
        human_idx_to_name = [elem for elem in dataset.data['scans'] if elem['take_idx'] == int(take_idx) and elem['scan'] == pcd_idx][0]['human_idx_to_name']
        human_name_to_idx = {value: key for key, value in human_idx_to_name.items()}
        # We will rename some things here.
        if 'circulating-nurse' in human_name_to_idx:
            human_name_to_idx['circulator'] = human_name_to_idx.pop('circulating-nurse')
        if 'head-surgeon' in human_name_to_idx:
            human_name_to_idx['head_surgeon'] = human_name_to_idx.pop('head-surgeon')
        if 'assistant-surgeon' in human_name_to_idx:
            human_name_to_idx['assistant_surgeon'] = human_name_to_idx.pop('assistant-surgeon')
        human_idx_to_name = {value: key for key, value in human_name_to_idx.items()}

        image_paths = load_image_paths(scan_id_no_split)
        objs = dataset.objs_json[scan_id]
        reverse_objs = {value: key for key, value in objs.items()}
        if 'Patient' in reverse_objs:
            reverse_objs['patient'] = reverse_objs.pop('Patient')
        objs = {value: key for key, value in reverse_objs.items()}
        relations = dataset.relationship_json[scan_id]
        relations = [(objs[sub_idx], rel_name, objs[obj_idx]) for (sub_idx, obj_idx, rel_idx, rel_name) in relations]

        image_paths = [image_paths[view_idx - 1] for view_idx in views_to_use]

        for permutation_idx in range(n_permutations):  # TODO does id need to be unique? because right now it is not
            shuffle(relations)  # order should be random
            scene_graph_string = scene_graph_to_string(relations, human_idx_to_name)
            sample = apply_template(image_paths, scene_graph_string)
            samples.append(sample)

    return samples


def main():
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
    # entities = ["head surgeon", "assistant surgeon", "circulator", "nurse", "anaesthetist", "patient", "instrument table", "operating table",
    #             "secondary table", "anesthesia equipment", "instrument"]
    # predicates = ["assisting", "cementing", "cleaning", "closeTo", "cutting", "drilling", "hammering", "holding", "lyingOn", "manipulating",
    #               "preparing", "sawing", "suturing", "touching"]
    # tokenizer.add_special_tokens({"additional_special_tokens": entities + predicates})

    train_dataset = ORDataset(config, 'train', shuffle_objs=True)
    val_dataset = ORDataset(config, 'val')

    train_samples = generate_finetuning_samples_from_dataset(train_dataset)
    # Load the tokenizer which will be used
    # val_samples = generate_finetuning_samples_from_dataset(val_dataset)
    # Also calculate the corresponding word frequencies
    token_freq = Counter()
    longest_sample = -1
    for sample in train_samples:
        for conversation in sample['conversations']:
            if conversation['from'] == 'gpt':
                tokenized = tokenizer.tokenize(conversation['value'])
                token_freq.update(tokenized)
                longest_sample = max(longest_sample, len(tokenized))

    with open(f'data/llava_samples/train_multiview.json', 'w') as f:
        json.dump(train_samples, f, indent=4)

    with open(f'data/llava_samples/train_token_freqs_7b.json', 'w') as f:
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
