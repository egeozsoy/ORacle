import random
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from random import shuffle

import transformers
from tqdm import tqdm

from scene_graph_prediction.llava_helpers.descriptors import ENTITY_DESCRIPTORS_TRAINING, PREDICATE_DESCRIPTORS_TRAINING, ENTITY_SYMBOLS, PREDICATE_SYMBOLS
from synthetic_or_generation.generate_novel_augmentations import replacement_map, EQUIPMENT, INSTRUMENTS

warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl


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


def apply_template(image_paths, scene_graph, timepoint, SYMBOLIC_SG_MAP=None, image_json=None):
    # human_prompt = 'Describe this image using a scene graph, represented as a list of triplets. Each triplet consists of a subject(entity), an object(entity), and a predicate. Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching].'
    if SYMBOLIC_SG_MAP is not None:
        # integrate the symbols and knowledge
        entity_symbol_to_descriptor_sorted = sorted(SYMBOLIC_SG_MAP["entity_symbol_to_descriptor"].items(), key=lambda x: x[0])
        predicate_symbol_to_descriptor_sorted = sorted(SYMBOLIC_SG_MAP["predicate_symbol_to_descriptor"].items(), key=lambda x: x[0])
        entity_symbols = ", ".join([elem[0] for elem in entity_symbol_to_descriptor_sorted])
        predicate_symbols = ", ".join([elem[0] for elem in predicate_symbol_to_descriptor_sorted])
        human_prompt = f'Entities: [{entity_symbols}]. Predicates: [{predicate_symbols}]. <knowledge_start> '
        for entity_symbol, descriptor in entity_symbol_to_descriptor_sorted:
            human_prompt += f'{entity_symbol}: {descriptor} '
        for predicate_symbol, descriptor in predicate_symbol_to_descriptor_sorted:
            human_prompt += f'{predicate_symbol}: {descriptor} '
        human_prompt += f'<knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
    else:
        human_prompt = 'Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
    id = f'{image_paths[0].parent.parent.stem}/{timepoint}'
    value = scene_graph
    sample = {'id': id, 'timepoint': timepoint, 'image': [str(image_path.absolute()) for image_path in image_paths] if len(image_paths) > 1 else str(image_paths[0].absolute()),
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
    if image_json['replaced_pred'] is not None:
        used_predicates.add(image_json['replaced_pred'].lower())

    # equipment without used_entities
    allowed_entities = EQUIPMENT - used_entities
    allowed_predicates = (PREDICATE_DESCRIPTORS_TRAINING.keys() | {elem['replace_pred_to'][0].lower() + elem['replace_pred_to'][1:] for elem in replacement_map.values() if
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


def generate_finetuning_samples(path, views_to_use=(2,), SYMBOLIC_SG=False):
    samples = []
    all_json_paths = list(path.glob('*.json'))
    shuffle(all_json_paths)
    json_path_to_json = {json_path: _get_image_json(json_path) for json_path in tqdm(all_json_paths, desc='Loading jsons attributes')}
    object_to_valid_attributes = defaultdict(list)
    for json_path in all_json_paths:
        image_json = json_path_to_json[json_path]
        for name, descs in image_json['descriptors'].items():
            if name.lower().replace('_', ' ') in EQUIPMENT:
                # entity
                name = name.lower().replace('_', ' ')
            else:
                # predicate
                name = name[0].lower() + name[1:]
            object_to_valid_attributes[name].append(descs)
    for json_path in all_json_paths:
        image_json = json_path_to_json[json_path]
        negative_descriptors = _sample_negatives(image_json=image_json, object_to_valid_attributes=object_to_valid_attributes, k_negative_entities=3, k_negative_predicates=3)

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
            visual_attributes = {}
            textual_attributes = {}
            entities_to_delete = []
            predicates_to_delete = []

            # we first remove the predicate we replaced
            if image_json['replaced_pred'] is not None:
                predicate_descriptors.pop(image_json['replaced_pred'].lower())

            for entity_name, descriptors in entity_descriptors.items():
                # sample the attributes we want to keep. Fake everything except ('object'type') (but only with a probility
                # descriptors can be a list or dict. If list, we skip here
                if isinstance(descriptors, list):
                    continue
                elif isinstance(descriptors, dict):
                    visual_attributes[entity_name] = descriptors
                    all_attributes = sorted(list(descriptors.items()))
                    n_attributes = random.randint(1, len(all_attributes))
                    picked_attributes = {key: value for key, value in random.sample(all_attributes, n_attributes)}
                    textual_attributes[entity_name] = picked_attributes
                    text_descriptor = ", ".join(picked_attributes.values()) + '.'
                    entity_descriptors[entity_name] = [text_descriptor]
                else:
                    raise NotImplementedError

            for predicate_name, descriptors in predicate_descriptors.items():
                # sample the attributes we want to keep. Fake everything except ('object'type') (but only with a probility
                # descriptors can be a list or dict. If list, we skip here
                if isinstance(descriptors, list):
                    continue
                elif isinstance(descriptors, dict):
                    visual_attributes[predicate_name] = descriptors
                    all_attributes = sorted(list(descriptors.items()))
                    n_attributes = random.randint(1, len(all_attributes))
                    picked_attributes = {key: value for key, value in random.sample(all_attributes, n_attributes)}
                    textual_attributes[predicate_name] = picked_attributes
                    text_descriptor = ", ".join(picked_attributes.values())
                    text_descriptor = f'use of a tool: {text_descriptor}.'
                    predicate_descriptors[predicate_name] = [text_descriptor]
                else:
                    raise NotImplementedError

            # We not only add the correct descriptors, but also some random descriptors.
            for all_descriptors, is_inscene in [(image_json['descriptors'], True)] + [(descriptor, False) for descriptor in negative_descriptors]:
                for name, descriptors in all_descriptors.items():
                    is_entity = descriptors['object_type'] in EQUIPMENT
                    is_predicate = descriptors['object_type'] in INSTRUMENTS
                    assert is_entity or is_predicate
                    all_attributes = sorted(list(descriptors.items()))
                    n_attributes = random.randint(1, len(all_attributes))
                    picked_attributes = {key: value for key, value in random.sample(all_attributes, n_attributes)}
                    if is_inscene:
                        if is_entity:
                            name_tmp = name.lower().replace('_', ' ')
                        if is_predicate:
                            name_tmp = name[0].lower() + name[1:]
                        visual_attributes[name_tmp] = descriptors
                        textual_attributes[name_tmp] = picked_attributes

                        if is_entity and name_tmp in entities_to_delete: entities_to_delete.remove(name_tmp)
                        if is_predicate and name_tmp in predicates_to_delete: predicates_to_delete.remove(name_tmp)

                    text_descriptor = ", ".join(picked_attributes.values())
                    # first letter should be capitalized. The sentence should end with a period.
                    if descriptors['object_type'] in INSTRUMENTS:
                        text_descriptor = f'use of a tool: {text_descriptor}.'
                        # first letter of name should be lowered
                        name = name[0].lower() + name[1:]
                        predicate_descriptors[name] = [text_descriptor]
                    elif descriptors['object_type'] in EQUIPMENT:
                        text_descriptor = text_descriptor + '.'
                        name = name.lower().replace('_', ' ')
                        entity_descriptors[name] = [text_descriptor]
                    else:
                        raise NotImplementedError

            for entity_name, descriptors in entity_descriptors.items():
                entity_name_to_symbol[entity_name] = entity_symbols.pop()
                entity_symbol_to_descriptor[entity_name_to_symbol[entity_name]] = random.choice(descriptors)
            for predicate_name, descriptors in predicate_descriptors.items():
                predicate_name_to_symbol[predicate_name] = predicate_symbols.pop()
                predicate_symbol_to_descriptor[predicate_name_to_symbol[predicate_name]] = random.choice(descriptors)

            # clean the scene graph by deleting the entities and predicates we don't want to use
            new_graph = []

            for (sub, rel, obj) in relations:
                if sub.replace('_', ' ') not in entities_to_delete and rel.lower() not in predicates_to_delete and obj.replace('_', ' ') not in entities_to_delete:
                    new_graph.append((sub, rel, obj))
            relations = new_graph

            symbolic_sg_map = {'entity_name_to_symbol': entity_name_to_symbol, 'entity_symbol_to_descriptor': entity_symbol_to_descriptor, 'predicate_name_to_symbol': predicate_name_to_symbol,
                               'predicate_symbol_to_descriptor': predicate_symbol_to_descriptor}
        else:
            symbolic_sg_map = None
        scene_graph_string = scene_graph_to_string(relations, SYMBOLIC_SG_MAP=symbolic_sg_map)
        sample = apply_template(image_paths, scene_graph_string, timepoint=int(json_path.stem), SYMBOLIC_SG_MAP=symbolic_sg_map, image_json=image_json)
        samples.append(sample)

    return samples


def main():
    SYMBOLIC_SG = True
    SPLIT = 'train'
    # views_to_use = (2,)
    views_to_use = (2, 1, 3, 5)
    if SYMBOLIC_SG:
        NAME = f'{SPLIT}_symbolic_synthetic'
    else:
        NAME = f'{SPLIT}_synthetic'
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
    if len(views_to_use) > 1:
        dataset_path = Path('/home/guests/shared/Oracle/synthetic_4D-OR_mv')
    else:
        dataset_path = Path('synthetic_or_generation/synthetic_4D-OR')

    samples = generate_finetuning_samples(dataset_path, views_to_use=views_to_use, SYMBOLIC_SG=SYMBOLIC_SG)
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

    with open(f'data/llava_samples/{NAME}.json', 'w') as f:
        json.dump(samples, f, indent=4)

    if SPLIT == 'train':
        if SYMBOLIC_SG:
            with open(f'data/llava_samples/train_token_freqs_7b_symbolic_synthetic.json', 'w') as f:
                json.dump(token_freq, f, indent=4)
        else:
            with open(f'data/llava_samples/train_token_freqs_7b_perm_synthetic.json', 'w') as f:
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
    main()
