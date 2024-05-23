import argparse
import os
import random
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from random import shuffle

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl
import transformers
from tqdm import tqdm

from LLaVA.llava.constants import VIS_DESCRIPTOR_TOKEN
from scene_graph_prediction.llava_helpers.descriptors import ENTITY_DESCRIPTORS_TRAINING, PREDICATE_DESCRIPTORS_TRAINING, ENTITY_SYMBOLS, \
    PREDICATE_SYMBOLS
from synthetic_or_generation.generate_novel_augmentations import replacement_map, EQUIPMENT, INSTRUMENTS

warnings.filterwarnings('ignore')


def scene_graph_to_string(scene_graph, SYMBOLIC_SG_MAP=None):
    '''
    Scene graph is a list of relations in the form of (subject, relation, object)
    '''
    out = '<SG> '
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

    out = out.rstrip('; ') + ' </SG>'
    return out


def apply_template(image_paths, scene_graph, timepoint, SYMBOLIC_SG_MAP=None, sym_to_descriptor_paths=None):
    # human_prompt = 'Describe this image using a scene graph, represented as a list of triplets. Each triplet consists of a subject(entity), an object(entity), and a predicate. Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching].'
    if SYMBOLIC_SG_MAP is not None:
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
                vis_knowledge_paths.extend(sym_to_descriptor_paths[entity_symbol])
        for predicate_symbol, descriptor in predicate_symbol_to_descriptor_sorted:
            human_prompt += f'{predicate_symbol}: {descriptor} '
            if predicate_symbol in sym_to_descriptor_paths:
                vis_knowledge_paths.extend(sym_to_descriptor_paths[predicate_symbol])
        human_prompt += f'<knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
    else:
        human_prompt = 'Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
    id = f'{image_paths[0].parent.parent.stem}/{timepoint}'
    value = scene_graph
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
                      "value": value
                  },
              ]
              }

    return sample


def _get_image_json(json_path):
    # with open(json_path, 'r') as f: return json.load(f)
    with open(json_path.with_suffix('.json'), 'r') as f: return json.load(f)


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


def generate_finetuning_samples(path, views_to_use=(2,), SYMBOLIC_SG=False):
    samples = []
    all_json_paths = list(path.glob('*.json'))
    shuffle(all_json_paths)
    json_path_to_json = {json_path: _get_image_json(json_path) for json_path in tqdm(all_json_paths, desc='Loading jsons attributes')}

    object_to_valid_attributes = defaultdict(list)
    paths_to_remove = []
    for json_path in tqdm(all_json_paths, desc='Loading jsons attributes'):
        image_json = json_path_to_json[json_path]
        skip = False
        for elem in image_json['paths']:
            if len(image_json['paths'][elem]) == 0:
                skip = True
                paths_to_remove.append(json_path)
                break
        if skip:
            continue
        for name, descs in image_json['descriptors'].items():
            if name.lower().replace('_', ' ') in EQUIPMENT:
                # entity
                name = name.lower().replace('_', ' ')
            else:
                # predicate
                name = name[0].lower() + name[1:]
            object_to_valid_attributes[name].append(descs)
    all_json_paths = [path for path in all_json_paths if path not in paths_to_remove]

    print(f'Loaded {len(all_json_paths)} jsons')

    all_visual_descriptors = list(Path('synthetic_or_generation/vis_descriptors').glob('*.pt'))
    object_to_descriptor_paths = defaultdict(list)
    object_view_to_descriptor_paths = defaultdict(lambda: defaultdict(list))

    for descriptor_path in tqdm(all_visual_descriptors, "Loading visual descriptors"):
        object_type = descriptor_path.stem.split('_')[0]
        object_to_descriptor_paths[object_type].append(descriptor_path.stem)
        if 'cidx' in descriptor_path.stem:
            object_view_to_descriptor_paths[object_type][int(descriptor_path.stem.split('cidx')[1][0])].append(descriptor_path.stem)

    or_objs_to_cidx = defaultdict(set)  # only for 4D-OR objects
    all_4dor_descriptors = list(Path('synthetic_or_generation/original_crops').glob('*.pt'))
    assert len(all_4dor_descriptors) > 0
    for descriptor_path in all_4dor_descriptors:
        object_type = descriptor_path.stem.split('_')[0]
        if 'cam' in descriptor_path.stem:
            cidx = int(descriptor_path.stem.split('cam')[1][0])
            or_objs_to_cidx[object_type].add(cidx)

    for json_path in tqdm(all_json_paths, desc='Generating samples'):
        image_json = json_path_to_json[json_path]
        image_descriptor_paths = {k.lower().replace('_', ' '): v for k, v in image_json['paths'].items()}
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

        image_paths = []
        if len(views_to_use) > 1:
            for view_idx in views_to_use:
                image_paths.append(json_path.parent / f'{json_path.stem}_cidx{view_idx}.jpg')
        else:
            image_paths.append(json_path.with_suffix('.jpg'))

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
                    if entity_name in image_descriptor_paths:  # synthetic object
                        cams = [2]
                        n_cams = 1
                        success = False
                        while not success:
                            success = True
                            crop_paths = []
                            for cidx in cams:
                                random_aug_nr = random.randint(0, 9)
                                crop_path = image_descriptor_paths[entity_name][0] + f"_crop_cidx{cidx}_aug_{random_aug_nr}.pt"
                                if not os.path.exists("synthetic_or_generation/vis_descriptors/" + crop_path):
                                    print("Could not find", crop_path)
                                    success = False
                                    break
                                else:
                                    crop_paths.append(crop_path)
                        crop_paths = ["synthetic_or_generation/vis_descriptors/" + crop_path for crop_path in crop_paths]
                    else:  # real 4D-OR object
                        cams = [2]
                        n_cams = 1
                        crop_paths = []
                        for cidx in cams:
                            random_aug_nr = random.randint(0, 99)
                            crop_path = f'synthetic_or_generation/original_crops/{entity_name}_cam{cidx}_aug_{random_aug_nr}.pt'
                            crop_paths.append(crop_path)

                    vis_descriptor_paths[entity_name] = crop_paths
                    text_descriptor = '' + f'{VIS_DESCRIPTOR_TOKEN} ' * n_cams
                    text_descriptor = text_descriptor[:-1] + '.'
                    entity_descriptors[entity_name] = [text_descriptor]
                else:
                    raise NotImplementedError

            for predicate_name, descriptors in predicate_descriptors.items():
                # sample the attributes we want to keep. Fake everything except ('object'type') (but only with a probility
                # descriptors can be a list or dict. If list, we skip here
                if isinstance(descriptors, list):
                    continue
                elif isinstance(descriptors, dict):
                    if predicate_name in image_descriptor_paths:  # synthetic object
                        cams = [2]
                        n_cams = 1
                        success = False
                        while not success:
                            success = True
                            crop_paths = []
                            for cidx in cams:
                                random_aug_nr = random.randint(0, 9)
                                crop_path = image_descriptor_paths[predicate_name][0] + f"_crop_cidx{cidx}_aug_{random_aug_nr}.pt"
                                if not os.path.exists("synthetic_or_generation/vis_descriptors/" + crop_path):
                                    print("Could not find", crop_path)
                                    success = False
                                    break
                                else:
                                    crop_paths.append(crop_path)
                        crop_paths = ["synthetic_or_generation/vis_descriptors/" + crop_path for crop_path in crop_paths]
                    else:  # real 4D-OR object
                        cams = [2]
                        n_cams = 1
                        crop_paths = []
                        for cidx in cams:
                            random_aug_nr = random.randint(0, 99)
                            crop_path = f'synthetic_or_generation/original_crops/{predicate_name}_cam{cidx}_aug_{random_aug_nr}.pt'
                            crop_paths.append(crop_path)
                    vis_descriptor_paths[predicate_name] = crop_paths
                    text_descriptor = f'use of a tool: ' + f'{VIS_DESCRIPTOR_TOKEN} ' * n_cams
                    text_descriptor = text_descriptor[:-1] + '.'
                    predicate_descriptors[predicate_name] = [text_descriptor]
                else:
                    raise NotImplementedError

            # We not only add the correct descriptors, but also some random descriptors. Until now we ignored the descriptors of synthetic objects not in original 4dor dataset
            for all_descriptors, is_inscene in [(image_json['descriptors'], True)] + [(descriptor, False) for descriptor in negative_descriptors]:
                for name, descriptors in all_descriptors.items():
                    is_entity = descriptors['object_type'] in EQUIPMENT
                    is_predicate = descriptors['object_type'] in INSTRUMENTS
                    assert is_entity or is_predicate
                    picked_attributes = descriptors

                    name = name.lower().replace('_', ' ')

                    if name in image_descriptor_paths:  # synthetic object in scene
                        if name in vis_descriptor_paths:  # we already took care of this object
                            continue
                        else:
                            cams = [2]
                            n_cams = 1
                            success = False
                            while not success:
                                success = True
                                crop_paths = []
                                for cidx in cams:
                                    random_aug_nr = random.randint(0, 9)
                                    crop_path = image_descriptor_paths[name][0] + f"_crop_cidx{cidx}_aug_{random_aug_nr}.pt"
                                    if not os.path.exists("synthetic_or_generation/vis_descriptors/" + crop_path):
                                        print("Could not find", crop_path)
                                        success = False
                                        break
                                    else:
                                        crop_paths.append(crop_path)
                            crop_paths = ["synthetic_or_generation/vis_descriptors/" + crop_path for crop_path in crop_paths]

                    else:  # image not in scene
                        assert is_inscene == False  # this iterates only through synthetic objects and negative samples
                        path_prefix = f'{picked_attributes["object_type"]}_{picked_attributes["color"]}_{picked_attributes["size"]}_{picked_attributes["shape"]}_{picked_attributes["texture"]}'
                        cams = [2]
                        n_cams = 1
                        success = False
                        while not success:
                            success = True
                            crop_paths = []
                            for cidx in cams:
                                try:
                                    crop_path = random.choice([path for path in object_view_to_descriptor_paths[picked_attributes["object_type"]][cidx] if path.startswith(path_prefix)]) + ".pt"
                                    if not os.path.exists("synthetic_or_generation/vis_descriptors/" + crop_path):
                                        print("Could not find", crop_path)
                                        success = False
                                        break
                                    else:
                                        crop_paths.append(crop_path)
                                except Exception as e:
                                    print(e)
                                    success = False
                                    break

                        crop_paths = ["synthetic_or_generation/vis_descriptors/" + crop_path for crop_path in crop_paths]
                    vis_descriptor_paths[name] = crop_paths
                    text_descriptor = '' + f'{VIS_DESCRIPTOR_TOKEN} ' * len(crop_paths)
                    text_descriptor = text_descriptor[:-1] + '.'
                    # first letter should be capitalized. The sentence should end with a period.
                    if descriptors['object_type'] in INSTRUMENTS:
                        # first letter of name should be lowered
                        name = name[0].lower() + name[1:]
                        predicate_descriptors[name] = [f'use of a tool: {text_descriptor}']
                    elif descriptors['object_type'] in EQUIPMENT:
                        name = name.lower().replace('_', ' ')
                        entity_descriptors[name] = [text_descriptor]

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
                new_graph.append((sub, rel, obj))
            relations = new_graph

            symbolic_sg_map = {'entity_name_to_symbol': entity_name_to_symbol, 'entity_symbol_to_descriptor': entity_symbol_to_descriptor,
                               'predicate_name_to_symbol': predicate_name_to_symbol,
                               'predicate_symbol_to_descriptor': predicate_symbol_to_descriptor}
        else:
            symbolic_sg_map = None
        scene_graph_string = scene_graph_to_string(relations, SYMBOLIC_SG_MAP=symbolic_sg_map)
        sample = apply_template(image_paths, scene_graph_string, timepoint=int(json_path.stem),
                                SYMBOLIC_SG_MAP=symbolic_sg_map, sym_to_descriptor_paths=sym_to_descriptor_paths)
        samples.append(sample)

    return samples


def main():
    SYMBOLIC_SG = True
    SPLIT = 'train'
    # views_to_use = (2)
    views_to_use = (2, 1, 3, 5)

    if SYMBOLIC_SG:
        NAME = f'{SPLIT}_symbolic_synthetic_visual'
    else:
        NAME = f'{SPLIT}_synthetic_visual'
    if len(views_to_use) > 1:
        NAME += f'_{len(views_to_use)}views'

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

    dataset_path = Path('synthetic_or_generation/synthetic_4D-OR_mv')

    samples = generate_finetuning_samples(dataset_path, views_to_use=views_to_use, SYMBOLIC_SG=SYMBOLIC_SG)
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


if __name__ == '__main__':
    main()
