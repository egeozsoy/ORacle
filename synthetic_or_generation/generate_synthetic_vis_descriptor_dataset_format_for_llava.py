import random
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from random import shuffle
import transformers
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from LLaVA.llava.constants import VIS_DESCRIPTOR_TOKEN
from helpers.configurations import OR_4D_DATA_ROOT_PATH
from scene_graph_prediction.llava_helpers.scene_graph_converters import extract_take_int_from_image_path, parse_llava_sg, llava_sg_to_surgery_sg, \
    surgery_sg_to_memory_str
from scene_graph_prediction.llava_helpers.descriptors import ENTITY_DESCRIPTORS_TRAINING, PREDICATE_DESCRIPTORS_TRAINING, ENTITY_SYMBOLS, \
    PREDICATE_SYMBOLS
from synthetic_or_generation.generate_novel_augmentations import replacement_map, EQUIPMENT, INSTRUMENTS
from synthetic_or_generation.generate_novel_entities import sample_attributes, surgical_tools_and_equipment, ATTRIBUTES

warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl

def scene_graph_to_string(scene_graph, SG_INDICATOR='double', SYMBOLIC_SG_MAP=None):
    '''
    Scene graph is a list of relations in the form of (subject, relation, object)
    '''
    if SG_INDICATOR == 'double':
        out = '<SG> '
    else:
        raise NotImplementedError
    for (subject, relation, object) in scene_graph:
        subject = subject.replace('_', ' ').lower()
        object = object.replace('_', ' ').lower()
        if 'human' in subject:  # Default to circulator
            subject = 'circulator'
        if 'human' in object:  # Default to circulator
            object = 'circulator'
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

        out += f'{subject},{object},{relation}; '

    if SG_INDICATOR == 'double':
        # remove the last ";" and add the end token.
        out = out.rstrip('; ') + ' </SG>'
    return out


def apply_template(image_paths, scene_graph, image_idx, INCLUDE_TIMEPOINT=True, SYMBOLIC_SG_MAP=None, sym_to_descriptor_paths=None):
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
            human_prompt += f'{entity_symbol}: {descriptor} '
            if entity_symbol in sym_to_descriptor_paths:
                vis_knowledge_paths.append(sym_to_descriptor_paths[entity_symbol])
        for predicate_symbol, descriptor in predicate_symbol_to_descriptor_sorted:
            human_prompt += f'{predicate_symbol}: {descriptor} '
            if predicate_symbol in sym_to_descriptor_paths:
                vis_knowledge_paths.append(sym_to_descriptor_paths[predicate_symbol])
        human_prompt += f'<knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
    else:
        human_prompt = 'Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
    id = f'{image_paths[0].parent.parent.stem}/{image_paths[0].stem}'
    value = scene_graph
    sample = {'id': id, 'timepoint': image_idx,
              'image': [str(image_path.absolute()) for image_path in image_paths] if len(image_paths) > 1 else str(image_paths[0].absolute()),
              'vis_knowledge_paths': vis_knowledge_paths if SYMBOLIC_SG_MAP is not None else None,
              "conversations": [
                  {
                      "from": "human",
                      "value": f"<image>\n{human_prompt}"
                  },
                  {
                      "from": "gpt",
                      "value": value
                  },
              ]
              }

    return sample


def _get_image_json(image_path):
    with open(image_path.with_suffix('.json'), 'r') as f: return json.load(f)


def _sample_negatives(image_json, object_to_valid_attributes, k_negative_entities=3, k_negative_predicates=3):
    '''
    Negative entities or negative predicates can be anything that is not already in use in the scene, or will be in use after we modify the image.

    '''
    used_entities = set()
    used_predicates = set()  # should also include the replaced_pred, we also don't want that to be in the negative samples
    for (sub, rel, obj) in image_json['sg']:
        used_entities.add(sub.lower().replace('_', ' '))
        used_entities.add(obj.lower().replace('_', ' '))
        used_predicates.add(rel[0].lower() + rel[1:])
    if image_json['replaced_pred'] is not None:
        used_predicates.add(image_json['replaced_pred'].lower())

    # equipment without used_entities
    allowed_entities = EQUIPMENT - used_entities
    allowed_predicates = (PREDICATE_DESCRIPTORS_TRAINING.keys() | {elem['replace_pred_to'][0].lower() + elem['replace_pred_to'][1:] for elem in
                                                                   replacement_map.values() if
                                                                   'replace_pred_to' in elem}) - used_predicates

    negative_entity_samples = []
    negative_predicate_samples = []

    while len(negative_entity_samples) < k_negative_entities:
        random_object_type = random.choice(list(object_to_valid_attributes.keys()))
        attributes = random.choice(object_to_valid_attributes[random_object_type])
        if attributes['object_type'] in allowed_entities:
            negative_entity_samples.append({attributes['object_type']: attributes})

    while len(negative_predicate_samples) < k_negative_predicates:
        random_object_type = random.choice(list(object_to_valid_attributes.keys()))
        attributes = random.choice(object_to_valid_attributes[random_object_type])
        try:
            corresponding_predicate = replacement_map[attributes['object_type']]['replace_pred_to']
        except Exception as e:
            continue
        corresponding_predicate = corresponding_predicate[0].lower() + corresponding_predicate[1:]
        if corresponding_predicate in allowed_predicates:
            negative_predicate_samples.append({corresponding_predicate: attributes})

    return negative_entity_samples + negative_predicate_samples


def _fake_attributes(FAKE_P, object_to_valid_attributes, name, picked_attributes):
    if random.random() < FAKE_P:  # decide if the object should have real or fake attributes
        new_attributes = random.choice(object_to_valid_attributes[name])
        # check if any attribute actually changed, otherwise return None
        changed = False
        for key, value in picked_attributes.items():
            if new_attributes[key] != value:
                changed = True
            picked_attributes[key] = new_attributes[key]
        if not changed:
            return None
        return name

    return None

def generate_finetuning_samples(path, views_to_use=(2,), SG_INDICATOR='double', INCLUDE_TIMEPOINT=True, SYMBOLIC_SG=False, FAKE_ATTRIBUTES=False,
                                FAKE_P=0.2, CROP_AUGS = False):
    samples = []
    all_image_paths = list(path.glob('*.jpg'))
    shuffle(all_image_paths)
    image_path_to_json = {image_path: _get_image_json(image_path) for image_path in tqdm(all_image_paths, desc='Loading jsons attributes')}
    object_to_valid_attributes = defaultdict(list)
    for image_path in all_image_paths:
        image_json = image_path_to_json[image_path]
        for name, descs in image_json['descriptors'].items():
            if name.lower().replace('_', ' ') in EQUIPMENT:
                # entity
                name = name.lower().replace('_', ' ')
            else:
                # predicate
                name = name[0].lower() + name[1:]
            object_to_valid_attributes[name].append(descs)

    all_visual_descriptors = list(Path('synthetic_or_generation/vis_descriptors').glob('*.pt'))
    object_to_descriptor_paths = defaultdict(list)
    for descriptor_path in all_visual_descriptors:
        object_type = descriptor_path.stem.split('_')[0]
        object_to_descriptor_paths[object_type].append(descriptor_path.stem)

    for image_path in tqdm(all_image_paths):
        image_json = image_path_to_json[image_path]
        image_descriptor_paths = {k.lower().replace('_',' '):v for k,v in image_json['paths'].items()}
        negative_descriptors = _sample_negatives(image_json=image_json, object_to_valid_attributes=object_to_valid_attributes, k_negative_entities=3,
                                                 k_negative_predicates=3)

        relations = image_json['sg']
        new_relations = []
        for (sub, rel, obj) in relations:
            if 'circulating-nurse' in sub:
                sub = 'circulator'
            if 'circulating-nurse' in obj:
                obj = 'circulator'
            if 'head-surgeon' in sub:
                sub = 'head_surgeon'
            if 'head-surgeon' in obj:
                obj = 'head_surgeon'
            if 'assistant-surgeon' in sub:
                sub = 'assistant_surgeon'
            if 'assistant-surgeon' in obj:
                obj = 'assistant_surgeon'
            new_relations.append((sub, rel, obj))
        relations = new_relations

        image_paths = [image_path]

        shuffle(relations)  # order should be random
        if SYMBOLIC_SG:
            # we create a symbolic map here. Symbolic map first comes up with a random permutation of entity and predicate symbols. Then it matches every real entity and predicate name with a random symbol.
            # Also for every real entity and predicate name, it samples a random descriptor from the list of descriptors.
            entity_name_to_symbol = {}
            entity_symbol_to_descriptor = {}
            predicate_name_to_symbol = {}
            predicate_symbol_to_descriptor = {}
            entity_symbols = deepcopy(ENTITY_SYMBOLS)
            predicate_symbols = deepcopy(PREDICATE_SYMBOLS)
            shuffle(entity_symbols)
            shuffle(predicate_symbols)
            entity_descriptors = deepcopy(ENTITY_DESCRIPTORS_TRAINING)
            predicate_descriptors = deepcopy(PREDICATE_DESCRIPTORS_TRAINING)
            entities_to_delete = []
            predicates_to_delete = []
            vis_descriptor_paths = {}
            sym_to_descriptor_paths = {}
            # we first remove the predicate we replaced
            if image_json['replaced_pred'] is not None:
                predicate_descriptors.pop(image_json['replaced_pred'].lower())

            for entity_name, descriptors in entity_descriptors.items():
                # sample the attributes we want to keep. Fake everything except ('object'type') (but only with a probility
                # descriptors can be a list or dict. If list, we skip here
                if isinstance(descriptors, list):
                    continue
                elif isinstance(descriptors, dict):
                    picked_attributes = descriptors
                    if FAKE_ATTRIBUTES:
                        entity_to_delete = _fake_attributes(FAKE_P, object_to_valid_attributes, entity_name, picked_attributes)
                        if entity_to_delete is not None: entities_to_delete.append(entity_to_delete)
                    else:
                        entity_to_delete = None
                    if entity_name in image_descriptor_paths: #synthetic object
                        if entity_name == entity_to_delete: #object got fake attributes -> choose random crop with fitting attributes
                            path_prefix = f'{picked_attributes["object_type"]}_{picked_attributes["color"]}_{picked_attributes["size"]}_{picked_attributes["shape"]}_{picked_attributes["texture"]}'
                            if CROP_AUGS:
                                crop_path = random.choice([path for path in object_to_descriptor_paths[entity_name] if path.startswith(path_prefix)]) +".pt"
                            else:
                                crop_path = random.choice([path for path in object_to_descriptor_paths[entity_name] if path.startswith(path_prefix) and "_aug_" not in path]) + ".pt"
                            crop_path = "synthetic_or_generation/vis_descriptors/" + crop_path
                        else:
                            if CROP_AUGS:
                                random_aug_nr = random.randint(0, 9)
                                crop_path = image_descriptor_paths[entity_name] + f"_crop_aug_{random_aug_nr}.pt"
                            else:
                                crop_path = image_descriptor_paths[entity_name] + "_crop.pt"
                            crop_path = "synthetic_or_generation/vis_descriptors/" + crop_path
                    else: # real 4D-OR object
                        if entity_name == entity_to_delete: #object got fake attributes -> choose random crop with fitting attributes
                            path_prefix = f'{picked_attributes["object_type"]}_{picked_attributes["color"]}_{picked_attributes["size"]}_{picked_attributes["shape"]}_{picked_attributes["texture"]}'
                            if CROP_AUGS:
                                crop_path = random.choice([path for path in object_to_descriptor_paths[entity_name] if path.startswith(path_prefix)]) + ".pt"
                            else:
                                crop_path = random.choice([path for path in object_to_descriptor_paths[entity_name] if path.startswith(path_prefix) and "_aug_" not in path]) + ".pt"
                            crop_path = "synthetic_or_generation/vis_descriptors/" + crop_path
                        else:
                            if CROP_AUGS:
                                random_aug_nr = random.randint(0, 99)
                                crop_path = f'data/original_crops/{entity_name}_take1_aug_{random_aug_nr}.pt'
                            else:
                                crop_path = f'data/original_crops/{entity_name}_take1.pt'
                    vis_descriptor_paths[entity_name] = crop_path
                    text_descriptor = f'{VIS_DESCRIPTOR_TOKEN}.'
                    entity_descriptors[entity_name] = [text_descriptor]
                else:
                    raise NotImplementedError

            for predicate_name, descriptors in predicate_descriptors.items():
                # sample the attributes we want to keep. Fake everything except ('object'type') (but only with a probility
                # descriptors can be a list or dict. If list, we skip here
                if isinstance(descriptors, list):
                    continue
                elif isinstance(descriptors, dict):
                    picked_attributes = descriptors
                    if FAKE_ATTRIBUTES:
                        predicate_to_delete = _fake_attributes(FAKE_P, object_to_valid_attributes, predicate_name, picked_attributes)
                        if predicate_to_delete is not None: predicates_to_delete.append(predicate_to_delete)
                    else:
                        predicate_to_delete = None
                    if predicate_name in image_descriptor_paths: #synthetic object
                        if predicate_name == predicate_to_delete: #object got fake attributes -> choose random crop with fitting attributes
                            path_prefix = f'{picked_attributes["object_type"]}_{picked_attributes["color"]}_{picked_attributes["size"]}_{picked_attributes["shape"]}_{picked_attributes["texture"]}'
                            if CROP_AUGS:
                                crop_path = random.choice([path for path in object_to_descriptor_paths[picked_attributes["object_type"]] if path.startswith(path_prefix)]) + ".pt"
                            else:
                                crop_path = random.choice([path for path in object_to_descriptor_paths[picked_attributes["object_type"]] if path.startswith(path_prefix) and "_aug_" not in path]) + ".pt"
                            crop_path = "synthetic_or_generation/vis_descriptors/" + crop_path
                        else:
                            if CROP_AUGS:
                                random_aug_nr = random.randint(0, 9)
                                crop_path = image_descriptor_paths[predicate_name] + f"_crop_aug_{random_aug_nr}.pt"
                            else:
                                crop_path = image_descriptor_paths[predicate_name] + "_crop.pt"
                            crop_path = "synthetic_or_generation/vis_descriptors/" + crop_path
                    else: # real 4D-OR object
                        if predicate_name == predicate_to_delete: #object got fake attributes -> choose random crop with fitting attributes
                            path_prefix = f'{picked_attributes["object_type"]}_{picked_attributes["color"]}_{picked_attributes["size"]}_{picked_attributes["shape"]}_{picked_attributes["texture"]}'
                            if CROP_AUGS:
                                crop_path = random.choice([path for path in object_to_descriptor_paths[picked_attributes["object_type"]] if path.startswith(path_prefix)]) + ".pt"
                            else:
                                crop_path = random.choice([path for path in object_to_descriptor_paths[picked_attributes["object_type"]] if path.startswith(path_prefix) and "_aug_" not in path]) + ".pt"
                            crop_path = "synthetic_or_generation/vis_descriptors/" + crop_path
                        else:
                            if CROP_AUGS:
                                random_aug_nr = random.randint(0, 99)
                                crop_path = f'data/original_crops/{predicate_name}_take1_aug_{random_aug_nr}.pt'
                            else:
                                crop_path = f'data/original_crops/{predicate_name}_take1.pt'
                    vis_descriptor_paths[predicate_name] = crop_path
                    text_descriptor = f'use of a tool: {VIS_DESCRIPTOR_TOKEN}.'
                    predicate_descriptors[predicate_name] = [text_descriptor]
                else:
                    raise NotImplementedError

            # We not only add the correct descriptors, but also some random descriptors.
            for all_descriptors, is_inscene in [(image_json['descriptors'], True)] + [(descriptor, False) for descriptor in negative_descriptors]:
                for name, descriptors in all_descriptors.items():
                    is_entity = descriptors['object_type'] in EQUIPMENT
                    is_predicate = descriptors['object_type'] in INSTRUMENTS
                    assert is_entity or is_predicate
                    picked_attributes = descriptors
                    if is_inscene:
                        if is_entity:
                            name_tmp = name.lower().replace('_', ' ')
                        if is_predicate:
                            name_tmp = name[0].lower() + name[1:]
                        if FAKE_ATTRIBUTES:
                            to_delete = _fake_attributes(FAKE_P, object_to_valid_attributes, name_tmp, picked_attributes)
                        else:
                            to_delete = None
                        if to_delete is not None:
                            if is_entity: entities_to_delete.append(to_delete)
                            if is_predicate: predicates_to_delete.append(to_delete)
                        else:  # it might be already marked for deletion, then reverse that
                            if is_entity and name_tmp in entities_to_delete: entities_to_delete.remove(name_tmp)
                            if is_predicate and name_tmp in predicates_to_delete: predicates_to_delete.remove(name_tmp)

                    text_descriptor = f'{VIS_DESCRIPTOR_TOKEN}.'
                    # first letter should be capitalized. The sentence should end with a period.
                    if descriptors['object_type'] in INSTRUMENTS:
                        # first letter of name should be lowered
                        name = name[0].lower() + name[1:]
                        predicate_descriptors[name] = [f'use of a tool: {text_descriptor}']
                    elif descriptors['object_type'] in EQUIPMENT:
                        name = name.lower().replace('_', ' ')
                        entity_descriptors[name] = [text_descriptor]
                    else:
                        raise NotImplementedError

                    if name in image_descriptor_paths: #synthetic object
                        if is_inscene and name == to_delete: #object got fake attributes -> choose random crop with fitting attributes
                            path_prefix = f'{picked_attributes["object_type"]}_{picked_attributes["color"]}_{picked_attributes["size"]}_{picked_attributes["shape"]}_{picked_attributes["texture"]}'
                            if CROP_AUGS:
                                crop_path = random.choice([path for path in object_to_descriptor_paths[picked_attributes["object_type"]] if path.startswith(path_prefix)]) + ".pt"
                            else:
                                crop_path = random.choice([path for path in object_to_descriptor_paths[picked_attributes["object_type"]] if path.startswith(path_prefix) and "_aug_" not in path]) + ".pt"
                            crop_path = "synthetic_or_generation/vis_descriptors/" + crop_path
                        else:
                            if CROP_AUGS:
                                random_aug_nr = random.randint(0, 9)
                                crop_path = image_descriptor_paths[name] + f"_crop_aug_{random_aug_nr}.pt"
                            else:
                                crop_path = image_descriptor_paths[name] + "_crop.pt"
                            crop_path = "synthetic_or_generation/vis_descriptors/" + crop_path
                    else: # image not in scene
                        assert is_inscene == False # this iterates only through synthetic objects and negative samples
                        path_prefix = f'{picked_attributes["object_type"]}_{picked_attributes["color"]}_{picked_attributes["size"]}_{picked_attributes["shape"]}_{picked_attributes["texture"]}'
                        if CROP_AUGS:
                            crop_path = random.choice([path for path in object_to_descriptor_paths[picked_attributes["object_type"]] if path.startswith(path_prefix)]) + ".pt"
                        else:
                            crop_path = random.choice([path for path in object_to_descriptor_paths[picked_attributes["object_type"]] if path.startswith(path_prefix) and "_aug_" not in path]) + ".pt"
                        crop_path = "synthetic_or_generation/vis_descriptors/" + crop_path
                    vis_descriptor_paths[name] = crop_path


            for entity_name, descriptors in entity_descriptors.items():
                entity_name_to_symbol[entity_name] = entity_symbols.pop()
                entity_symbol_to_descriptor[entity_name_to_symbol[entity_name]] = random.choice(descriptors)
                if entity_name in vis_descriptor_paths:
                    sym_to_descriptor_paths[entity_name_to_symbol[entity_name]] = vis_descriptor_paths[entity_name]
            for predicate_name, descriptors in predicate_descriptors.items():
                predicate_name_to_symbol[predicate_name] = predicate_symbols.pop()
                predicate_symbol_to_descriptor[predicate_name_to_symbol[predicate_name]] = random.choice(descriptors)
                if predicate_name in vis_descriptor_paths:
                    sym_to_descriptor_paths[predicate_name_to_symbol[predicate_name]] = vis_descriptor_paths[predicate_name]

            # clean the scene graph by deleting the entities and predicates we don't want to use
            new_graph = []

            for (sub, rel, obj) in relations:
                if sub.replace('_', ' ') not in entities_to_delete and rel.lower() not in predicates_to_delete and obj.replace('_',
                                                                                                                               ' ') not in entities_to_delete:
                    new_graph.append((sub, rel, obj))
            relations = new_graph

            symbolic_sg_map = {'entity_name_to_symbol': entity_name_to_symbol, 'entity_symbol_to_descriptor': entity_symbol_to_descriptor,
                               'predicate_name_to_symbol': predicate_name_to_symbol,
                               'predicate_symbol_to_descriptor': predicate_symbol_to_descriptor}
        else:
            symbolic_sg_map = None
        scene_graph_string = scene_graph_to_string(relations, SG_INDICATOR=SG_INDICATOR, SYMBOLIC_SG_MAP=symbolic_sg_map)
        sample = apply_template(image_paths, scene_graph_string, image_idx=int(image_path.stem), INCLUDE_TIMEPOINT=INCLUDE_TIMEPOINT,
                                SYMBOLIC_SG_MAP=symbolic_sg_map, sym_to_descriptor_paths=sym_to_descriptor_paths)
        samples.append(sample)

    return samples


def main():
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
    FAKE_ATTRIBUTES = True
    CROP_AUGS = True
    FAKE_P = 0.2
    # TODO FLAG FOR TIMEPOINT and DROPPING. Naming Scheme should be so that only interesting flags are including in the name, not if they are False.
    if COMPACT_TEMPORAL:
        NAME = f'{SPLIT}_{ADD_TEMPORAL}temp_{MEMORY_INDICATOR}mem_{WITH_TEMPORAL_AUG}tempaug_{TEMPORAL_STYLE}_compact_{SG_INDICATOR}sg_synthetic'
    elif SYMBOLIC_SG:
        NAME = f'{SPLIT}_{ADD_TEMPORAL}temp_{MEMORY_INDICATOR}mem_{WITH_TEMPORAL_AUG}tempaug_{TEMPORAL_STYLE}_symbolic_{SG_INDICATOR}sg_synthetic_visual'
    else:
        NAME = f'{SPLIT}_{ADD_TEMPORAL}temp_{MEMORY_INDICATOR}mem_{WITH_TEMPORAL_AUG}tempaug_{TEMPORAL_STYLE}_{SG_INDICATOR}sg_synthetic'
    if FAKE_ATTRIBUTES:
        NAME += f'_fake_attributes_{FAKE_P}'
    if not INCLUDE_TIMEPOINT:
        NAME += '_notimepoints'
    if DROP_HISTORY is not False and DROP_HISTORY > 0.01:
        NAME += f'_drophistory{DROP_HISTORY}'
    if CROP_AUGS:
        NAME += '_cropaugs'

    print(f'Creating samples for LLAVA dataset with name {NAME}')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'liuhaotian/llava-v1.5-7b',
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )

    samples = generate_finetuning_samples(Path('synthetic_or_generation/synthetic_4D-OR'), SG_INDICATOR=SG_INDICATOR,
                                          INCLUDE_TIMEPOINT=INCLUDE_TIMEPOINT,
                                          SYMBOLIC_SG=SYMBOLIC_SG, FAKE_ATTRIBUTES=FAKE_ATTRIBUTES, FAKE_P=FAKE_P, CROP_AUGS=CROP_AUGS)
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
            memory_str2 = memory_str
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
            with open(f'data/llava_samples/train_token_freqs_7b_symbolic_synthetic_removal_{FAKE_P}_visual_noaug_negdesc.json', 'w') as f:
                json.dump(token_freq, f, indent=4)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    """format of json (ultimately):
    1) id
    2) image(s) paths
    3) Prompt (formulately in multiple ways)
    4) Answer (formulately in multiple ways) (multiple orders)
    5) Prompt should include knowledge about different things

    Optionally: include augmentations, modifications in scene graph, prompt or both etc.
    """
    main()