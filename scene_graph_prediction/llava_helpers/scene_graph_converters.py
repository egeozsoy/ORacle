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


def surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint):
    memory_str = ''
    for timepoint, (sub, pred,
                    obj) in surgery_sg_triplets:  # TODO actually intended was a different format. where we list per timepoint the changes. Then you would only have one T-N for each timepoint, then the scene graph.
        rel_timepoint = current_timepoint - timepoint
        # memory_str += f'{timepoint}: {sub},{obj},{pred}; '
        # instead we use relative timepoint with T-minus notation
        memory_str += f'T-{rel_timepoint}: {sub},{obj},{pred}; '

    if memory_str == '':
        return ''
    return memory_str[:-2]
