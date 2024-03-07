import random
import warnings
from collections import Counter
from random import shuffle
import transformers

from LLaVA.llava.constants import VIS_DESCRIPTOR_TOKEN
from helpers.configurations import OR_4D_DATA_ROOT_PATH
from scene_graph_prediction.llava_helpers.scene_graph_converters import extract_take_int_from_image_path, parse_llava_sg, llava_sg_to_surgery_sg, \
    surgery_sg_to_memory_str
from scene_graph_prediction.llava_helpers.descriptors import ENTITY_DESCRIPTORS, PREDICATE_DESCRIPTORS, ENTITY_SYMBOLS, PREDICATE_SYMBOLS

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


def scene_graph_to_string(scene_graph, human_idx_to_name, SG_INDICATOR='double', SYMBOLIC_SG_MAP=None):
    '''
    Scene graph is a list of relations in the form of (subject, relation, object)
    '''
    if SG_INDICATOR == 'double':
        out = '<SG> '
    else:
        raise NotImplementedError
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

        if SYMBOLIC_SG_MAP is not None:
            if subject == 'none' or object == 'none' or relation == 'none':
                continue
            subject = SYMBOLIC_SG_MAP["entity_name_to_symbol"][subject]
            object = SYMBOLIC_SG_MAP["entity_name_to_symbol"][object]
            relation = SYMBOLIC_SG_MAP["predicate_name_to_symbol"][relation]
        # if subject != 'head surgeon' or object != 'patient': # TODO remove this. This only uses the main action
        #     continue
        out += f'{subject},{object},{relation}; '

    # # Remove last comma, put end token
    # if out:
    #     out = out[:-2] + '.'
    # else:
    #     out = '.'  # empty scene graph
    if SG_INDICATOR == 'double':
        # remove the last ";" and add the end token.
        out = out.rstrip('; ') + ' </SG>'
    return out


def apply_template(image_paths, scene_graph, timepoint, INCLUDE_TIMEPOINT=True, SYMBOLIC_SG_MAP=None, sym_to_descriptor_paths=None):
    # human_prompt = 'Describe this image using a scene graph, represented as a list of triplets. Each triplet consists of a subject(entity), an object(entity), and a predicate. Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching].'
    if INCLUDE_TIMEPOINT:
        human_prompt = 'Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text. Do not include the timepoint format "T-" in the triplets.'
    elif SYMBOLIC_SG_MAP is not None:
        # integrate the symbols and knowledge
        vis_knowledge_paths = []
        entity_symbol_to_descriptor_sorted = sorted(SYMBOLIC_SG_MAP["entity_symbol_to_descriptor"].items(), key=lambda x: x[0])
        predicate_symbol_to_descriptor_sorted = sorted(SYMBOLIC_SG_MAP["predicate_symbol_to_descriptor"].items(), key=lambda x: x[0])
        entity_symbols = ", ".join([elem[0] for elem in entity_symbol_to_descriptor_sorted])
        predicate_symbols = ", ".join([elem[0] for elem in predicate_symbol_to_descriptor_sorted])
        human_prompt = f'Entities: [{entity_symbols}]. Predicates: [{predicate_symbols}]. <knowledge_start> '
        for entity_symbol, descriptor in entity_symbol_to_descriptor_sorted:
            if entity_symbol in sym_to_descriptor_paths:
                human_prompt += f'{entity_symbol}: {VIS_DESCRIPTOR_TOKEN}. '
                vis_knowledge_paths.append(sym_to_descriptor_paths[entity_symbol])
            else:
                human_prompt += f'{entity_symbol}: {descriptor} '
        for predicate_symbol, descriptor in predicate_symbol_to_descriptor_sorted:
            if predicate_symbol in sym_to_descriptor_paths:
                human_prompt += f'{predicate_symbol}: use of a tool: {VIS_DESCRIPTOR_TOKEN}. '
                vis_knowledge_paths.append(sym_to_descriptor_paths[predicate_symbol])
            else:
                human_prompt += f'{predicate_symbol}: {descriptor} '
        human_prompt += f'<knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
    else:
        human_prompt = 'Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
    # TODO potentially modify the input to say <SG>
    id = f'{image_paths[0].parent.parent.stem}/{image_paths[0].stem}'

    sample = {'id': id, 'timepoint': timepoint,
              'image': [str(image_path.absolute()) for image_path in image_paths] if len(image_paths) > 1 else str(image_paths[0].absolute()),
              'vis_knowledge_paths': vis_knowledge_paths if SYMBOLIC_SG_MAP is not None else None,
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


def generate_finetuning_samples_from_dataset(dataset, n_permutations=1, views_to_use=(2,), SG_INDICATOR='double', INCLUDE_TIMEPOINT=True,
                                             SYMBOLIC_SG=False):
    samples = []
    for index in range(len(dataset)):
        scan_id = dataset.scans[index]
        scan_id_no_split = scan_id.rsplit('_', 1)[0]
        take_idx, pcd_idx = scan_id_no_split.split('_')
        human_idx_to_name = [elem for elem in dataset.data['scans'] if elem['take_idx'] == int(take_idx) and elem['scan'] == pcd_idx][0][
            'human_idx_to_name']
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
            if SYMBOLIC_SG:
                # we create a symbolic map here. Symbolic map first comes up with a random permutation of entity and predicate symbols. Then it matches every real entity and predicate name with a random symbol.
                # Also for every real entity and predicate name, it samples a random descriptor from the list of descriptors.
                entity_name_to_symbol = {}
                entity_symbol_to_descriptor = {}
                predicate_name_to_symbol = {}
                predicate_symbol_to_descriptor = {}
                entity_symbols = ENTITY_SYMBOLS.copy()
                predicate_symbols = PREDICATE_SYMBOLS.copy()
                sym_to_descriptor_paths = {}
                shuffle(entity_symbols)
                shuffle(predicate_symbols)
                for entity_name, descriptors in ENTITY_DESCRIPTORS.items():
                    entity_name_to_symbol[entity_name] = entity_symbols.pop()
                    entity_symbol_to_descriptor[entity_name_to_symbol[entity_name]] = random.choice(descriptors)
                    crop_path = f'data/original_crops/{entity_name}_take1.pt'
                    # check if file exists
                    if Path(crop_path).exists():
                        sym_to_descriptor_paths[entity_name_to_symbol[entity_name]] = crop_path
                for predicate_name, descriptors in PREDICATE_DESCRIPTORS.items():
                    predicate_name_to_symbol[predicate_name] = predicate_symbols.pop()
                    predicate_symbol_to_descriptor[predicate_name_to_symbol[predicate_name]] = random.choice(descriptors)
                    crop_path = f'data/original_crops/{predicate_name}_take1.pt'
                    if Path(crop_path).exists():
                        sym_to_descriptor_paths[predicate_name_to_symbol[predicate_name]] = crop_path

                symbolic_sg_map = {'entity_name_to_symbol': entity_name_to_symbol, 'entity_symbol_to_descriptor': entity_symbol_to_descriptor,
                                   'predicate_name_to_symbol': predicate_name_to_symbol,
                                   'predicate_symbol_to_descriptor': predicate_symbol_to_descriptor}
            else:
                symbolic_sg_map = None
            scene_graph_string = scene_graph_to_string(relations, human_idx_to_name, SG_INDICATOR=SG_INDICATOR, SYMBOLIC_SG_MAP=symbolic_sg_map)
            sample = apply_template(image_paths, scene_graph_string, timepoint=int(pcd_idx), INCLUDE_TIMEPOINT=INCLUDE_TIMEPOINT,
                                    SYMBOLIC_SG_MAP=symbolic_sg_map, sym_to_descriptor_paths=sym_to_descriptor_paths)
            samples.append(sample)

    return samples


def main():
    N_PERM = 20
    ADD_TEMPORAL = False
    WITH_TEMPORAL_AUG = False
    MEMORY_INDICATOR = 'double'  # single: Memory, double: <memory_start> and <memory_end>
    TEMPORAL_STYLE = 'longshort'  # can be longshort or all or longshort_compact
    COMPACT_TEMPORAL = False
    INCLUDE_TIMEPOINT = False
    DROP_HISTORY = 0.5  # either False or float
    SG_INDICATOR = 'double'  # double: <SG> and </SG>
    SYMBOLIC_SG = True
    SPLIT = 'train'
    views_to_use = (2,)
    # TODO FLAG FOR TIMEPOINT and DROPPING. Naming Scheme should be so that only interesting flags are including in the name, not if they are False.
    if COMPACT_TEMPORAL:
        NAME = f'{SPLIT}_{N_PERM}perm_{ADD_TEMPORAL}temp_{MEMORY_INDICATOR}mem_{WITH_TEMPORAL_AUG}tempaug_{TEMPORAL_STYLE}_compact_{SG_INDICATOR}sg'
    elif SYMBOLIC_SG:
        NAME = f'{SPLIT}_{N_PERM}perm_{ADD_TEMPORAL}temp_{MEMORY_INDICATOR}mem_{WITH_TEMPORAL_AUG}tempaug_{TEMPORAL_STYLE}_symbolic_{SG_INDICATOR}sg_visual'
    else:
        NAME = f'{SPLIT}_{N_PERM}perm_{ADD_TEMPORAL}temp_{MEMORY_INDICATOR}mem_{WITH_TEMPORAL_AUG}tempaug_{TEMPORAL_STYLE}_{SG_INDICATOR}sg_{len(views_to_use)}view'
    if not INCLUDE_TIMEPOINT:
        NAME += '_notimepoints'
    if DROP_HISTORY is not False and DROP_HISTORY > 0.01:
        NAME += f'_drophistory{DROP_HISTORY}'
    print(f'Creating samples for LLAVA dataset with name {NAME}')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='oracle.json', help='configuration file name. Relative path under given path')
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

    samples = generate_finetuning_samples_from_dataset(dataset, n_permutations=N_PERM, SG_INDICATOR=SG_INDICATOR, INCLUDE_TIMEPOINT=INCLUDE_TIMEPOINT,
                                                       SYMBOLIC_SG=SYMBOLIC_SG, views_to_use=views_to_use)
    # Load the tokenizer which will be used
    # val_samples = generate_finetuning_samples_from_dataset(val_dataset)
    # Also calculate the corresponding word frequencies
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

    if ADD_TEMPORAL:
        print('Adding temporal information...')
        take_to_history = {}
        take_timepoint_to_memory_str = {}

        for take_int in range(1, 11):
            take_scene_graphs = [elem for elem in samples if extract_take_int_from_image_path(elem['image']) == take_int]
            # make unique
            take_scene_graphs = list({elem['image']: elem for elem in take_scene_graphs}.values())
            # sort by image_path
            take_scene_graphs = sorted(take_scene_graphs, key=lambda x: x['image'])
            take_scene_graphs_reformatted = []
            for take_scene_graph in take_scene_graphs:
                scene_graph = parse_llava_sg(take_scene_graph['conversations'][1]['value'])
                take_scene_graphs_reformatted.append({'timepoint_idx': take_scene_graph['timepoint'], 'scene_graph': scene_graph})
            if COMPACT_TEMPORAL:
                surgery_sg_triplets = llava_sg_to_surgery_sg(take_scene_graphs_reformatted, entity_of_interest='patient',
                                                             IRRELEVANT_PREDS=['closeto', 'closeTo', 'holding', 'touching'])
            else:
                surgery_sg_triplets = llava_sg_to_surgery_sg(take_scene_graphs_reformatted, entity_of_interest=None,
                                                             IRRELEVANT_PREDS=['closeto', 'closeTo'])
            with open(f'data/llava_samples/surgery_sg_{take_int}.json', 'w') as f:
                json.dump(surgery_sg_triplets, f)
            take_to_history[take_int] = surgery_sg_triplets

        llava_scene_graphs_with_history = []
        for llava_scene_graph in samples:
            image_path = llava_scene_graph['image']
            image_path = Path(image_path)
            take_int = extract_take_int_from_image_path(image_path)
            surgery_sg_triplets = take_to_history[take_int]
            timepoint = llava_scene_graph['timepoint']
            surgery_sg_triplets = [elem for elem in surgery_sg_triplets if elem[0] < timepoint]
            memory_str = surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint=timepoint, TEMPORAL_STYLE=TEMPORAL_STYLE,
                                                  COMPACT_TEMPORAL=COMPACT_TEMPORAL,
                                                  INCLUDE_TIMEPOINTS=INCLUDE_TIMEPOINT)
            take_timepoint_to_memory_str[f'{take_int}_{timepoint}'] = memory_str
            input = llava_scene_graph['conversations'][0]['value']

            if WITH_TEMPORAL_AUG:
                p = random.random()
                if p < 0.5:
                    memory_str = None
                elif p < 0.666:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint=timepoint, TEMPORAL_STYLE='short',
                                                          COMPACT_TEMPORAL=COMPACT_TEMPORAL,
                                                          INCLUDE_TIMEPOINTS=INCLUDE_TIMEPOINT, DROP_HISTORY=DROP_HISTORY)
                elif p < 0.833:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint=timepoint, TEMPORAL_STYLE='long',
                                                          COMPACT_TEMPORAL=COMPACT_TEMPORAL,
                                                          INCLUDE_TIMEPOINTS=INCLUDE_TIMEPOINT, DROP_HISTORY=DROP_HISTORY)
                else:
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint=timepoint, TEMPORAL_STYLE='longshort',
                                                          COMPACT_TEMPORAL=COMPACT_TEMPORAL,
                                                          INCLUDE_TIMEPOINTS=INCLUDE_TIMEPOINT, DROP_HISTORY=DROP_HISTORY)

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

        samples = llava_scene_graphs_with_history

        with open(f'data/llava_samples/{NAME}_take_timepoint_to_memory_str.json', 'w') as f:
            json.dump(take_timepoint_to_memory_str, f)

    with open(f'data/llava_samples/{NAME}.json', 'w') as f:
        json.dump(samples, f, indent=4)

    if SPLIT == 'train' and not ADD_TEMPORAL:
        if SYMBOLIC_SG:
            with open(f'data/llava_samples/train_token_freqs_7b_{N_PERM}perm_symbolic_visual.json', 'w') as f:
                json.dump(token_freq, f, indent=4)
        else:
            with open(f'data/llava_samples/train_token_freqs_7b_{N_PERM}perm.json', 'w') as f:
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