import json
import re
from collections import Counter
from pathlib import Path
from random import shuffle

from tqdm import tqdm

from scene_graph_prediction.scene_graph_helpers.dataset.dataset_utils import reversed_role_synonyms, reversed_synonyms

EOI_ORDER = ['patient', 'head surgeon', 'assistant surgeon', 'circulator', 'anaesthetist']
IRRELEVANT_PREDS = ['closeto', 'holding', 'closeTo']
PRED_COUNTER = Counter()


def collapse_sgs(sgs):
    '''
    We only take note of the changes. Given a timepoint, we want to collapse all the changes and know what the current status is. As an example if the we know from 100 timepoints ago that head surgeon is cutting patient and this did not change
    then we assume that this is still the case. Handle the case of stopping
    '''
    sub_obj_to_pred = {}  # key: (sub, obj), value: pred
    for timepoint_idx, (sub, pred, obj) in sgs:
        if pred.startswith('stopped '):
            if (sub, obj) in sub_obj_to_pred:
                del sub_obj_to_pred[(sub, obj)]
        else:
            sub_obj_to_pred[(sub, obj)] = pred

    return sub_obj_to_pred


def find_related_entities(scene_graph, entity_of_interest, multi_hop_n):
    def _find_related(current_entity, current_hop, visited):
        if current_hop > multi_hop_n:
            return set()

        visited.add(current_entity)
        related_entities = set()

        if current_hop == 0:
            related_entities.add(current_entity)  # Add the entity of interest at the start

        for sub, pred, obj in scene_graph:
            if sub == current_entity and obj not in visited:
                if current_hop < multi_hop_n:
                    related_entities.add(obj)
                    related_entities.update(_find_related(obj, current_hop + 1, visited.copy()))
            elif obj == current_entity and sub not in visited:
                if current_hop < multi_hop_n:
                    related_entities.add(sub)
                    related_entities.update(_find_related(sub, current_hop + 1, visited.copy()))

        return related_entities

    # Start with the entity of interest and an empty set for visited entities
    return _find_related(entity_of_interest, 0, set())


def llava_sg_to_surgery_sg(llava_sgs, entity_of_interest):
    '''
    Modifies the original function to only include changes that concern the specified entity.
    entity_of_interest: The entity to focus on (e.g., 'head surgeon').
    '''
    surgery_sg_triplets = []  # Records the timepoints as well as the corresponding change log.

    for elem in llava_sgs:
        sg = elem['scene_graph']
        timepoint = elem['timepoint_idx']
        prev_sg = collapse_sgs(surgery_sg_triplets)

        related_entities = find_related_entities(sg, entity_of_interest, multi_hop_n=0)  # 0 only entity of interest, 1 also related entities, 2 also related entities of related entities, etc.

        # Filter scene graph for changes involving the entity of interest
        current_sg = {(sub, obj): pred for (sub, pred, obj) in sg if
                      pred not in IRRELEVANT_PREDS and (sub == entity_of_interest or obj == entity_of_interest or sub in related_entities or obj in related_entities)}
        # Compare current_sg with previous (collapsed) sg. If there is a difference, add it to the modifications.
        additions = []
        removals = []
        for (sub, obj), pred in current_sg.items():
            if (sub, obj) not in prev_sg:
                additions.append((sub, pred, obj))
        for (sub, obj), pred in prev_sg.items():
            if (sub, obj) not in current_sg:
                removals.append((sub, pred, obj))
        modifications = []
        for sub, pred, obj in additions:
            PRED_COUNTER[pred] += 1
            modifications.append((timepoint, (sub, pred, obj)))
        for sub, pred, obj in removals:
            modifications.append((timepoint, (sub, f'stopped {pred}', obj)))
        shuffle(modifications)
        surgery_sg_triplets.extend(modifications)
    return surgery_sg_triplets


def extract_take_int_from_image_path(image_path):
    return int(re.findall('take(\d+)', str(image_path))[0])


def parse_llava_sg(llava_sg):
    triplet_str = llava_sg.split(';')
    triplets = []
    for triplet in triplet_str:
        triplet = triplet.replace('.', '').replace('</s>', '').replace('<s>', '').strip()
        if triplet == '':
            continue
        triplet = triplet.split(',')
        triplet = [elem.strip() for elem in triplet]
        if len(triplet) != 3:
            continue
        sub, obj, pred = triplet
        triplets.append((sub, pred, obj))
    return triplets


def surgery_sg_to_memory_str(surgery_sg_triplets):
    memory_str = ''
    for timepoint, (sub, pred, obj) in surgery_sg_triplets:
        memory_str += f'{timepoint}: {sub},{pred},{obj}; '

    if memory_str == '':
        return ''
    return memory_str[:-2]


def main():
    train_file_path = Path('data/llava_samples/train_25perm.json')
    with train_file_path.open('r') as f:
        llava_scene_graphs = json.load(f)
    take_to_history = {}
    take_timepoint_to_memory_str = {}
    for take_int in range(1, 11):
        take_scene_graphs = [elem for elem in llava_scene_graphs if extract_take_int_from_image_path(elem['image']) == take_int]
        # make unique
        take_scene_graphs = list({elem['image']: elem for elem in take_scene_graphs}.values())
        # sort by image_path
        take_scene_graphs = sorted(take_scene_graphs, key=lambda x: x['image'])
        take_scene_graphs_reformatted = []
        for take_scene_graph in take_scene_graphs:
            scene_graph = parse_llava_sg(take_scene_graph['conversations'][1]['value'])
            take_scene_graphs_reformatted.append({'timepoint_idx': take_scene_graph['timepoint'], 'scene_graph': scene_graph})

        surgery_sg_triplets = llava_sg_to_surgery_sg(take_scene_graphs_reformatted, entity_of_interest='patient')
        with open(f'data/llava_samples/surgery_sg_{take_int}.json', 'w') as f:
            json.dump(surgery_sg_triplets, f)
        take_to_history[take_int] = surgery_sg_triplets

    # produce a new file with _history appended to the name
    train_file_with_history_path = train_file_path.parent / f'{train_file_path.stem}_history{train_file_path.suffix}'
    take_timepoint_to_memory_str_path = train_file_path.parent / f'{train_file_path.stem}_take_timepoint_to_memory_str.json'
    llava_scene_graphs_with_history = []
    for llava_scene_graph in tqdm(llava_scene_graphs, desc='Adding history to LLAVA scene graphs'):
        image_path = llava_scene_graph['image']
        image_path = Path(image_path)
        take_int = extract_take_int_from_image_path(image_path)
        surgery_sg_triplets = take_to_history[take_int]
        timepoint = llava_scene_graph['timepoint']
        surgery_sg_triplets = [elem for elem in surgery_sg_triplets if elem[0] < timepoint]
        memory_str = surgery_sg_to_memory_str(surgery_sg_triplets)
        take_timepoint_to_memory_str[f'{take_int}_{timepoint}'] = memory_str
        input = llava_scene_graph['conversations'][0]['value']
        input = input.replace('<image>\n', f'<image>\nMemory: {memory_str}.')
        llava_scene_graph['conversations'][0]['value'] = input
        llava_scene_graphs_with_history.append(llava_scene_graph)

    # TODO for each timepoint in each surgery export the following: Previous Surgery SG, Current SG, Current Surgery SG (which is the correct merge of the previous and current SG, by deciding to add something to the surgery SG (can be also a stopped action removing something)
    # TODO optionally enchance training data for each timepoint by adding previous Surgery SG to them.

    with train_file_with_history_path.open('w') as f:
        json.dump(llava_scene_graphs_with_history, f)

    with take_timepoint_to_memory_str_path.open('w') as f:
        json.dump(take_timepoint_to_memory_str, f)


if __name__ == '__main__':
    main()
