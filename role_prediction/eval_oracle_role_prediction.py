import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from sklearn.metrics import classification_report

from helpers.configurations import TAKE_SPLIT
from scene_graph_prediction.scene_graph_helpers.dataset.dataset_utils import map_scene_graph_name_to_vocab_idx, map_vocab_idx_to_scene_graph_name


def compute_optimal_human_indices(pred: List[Tuple[int, int, int]], gt: torch.Tensor):
    '''
    Matches the human indices predicted by the model(NOT THE PATIENT) to the ground truth to lead the highest accuracy
    Human indices: [7,8,9,10,11] + 6 for the patient
    To simplify:
    - If GT does not include a human index, don't assign this in the predictions
    - (If PT does not include a human index, don't iterate over it)
    '''
    if (len(gt) == 0 or len(pred) == 0):
        return {}
    gt_human_indices = set(gt[torch.logical_and(gt >= 6, gt <= 11)].tolist())  # TODO >=6 because of the patient
    pred_argmax = torch.tensor(pred).flatten()  # In eval, it is already argmaxed
    pred_human_indices = set(pred_argmax[torch.logical_and(pred_argmax >= 6,
                                                           pred_argmax <= 11)].tolist())  # technically, pred is soft so it is making predictions for all of them,therefore we use the argmax values

    all_human_indices = sorted(gt_human_indices.union(pred_human_indices))
    index_permutations = list(itertools.permutations(all_human_indices))
    max_acc = -1
    optimal_human_index_map = None

    for index_permutation in index_permutations:
        human_index_map = {idx: idx_perm for idx, idx_perm in zip(all_human_indices, index_permutation)}
        gt_rels = []
        pred_rels = []
        for gt_sub, gt_rel, gt_obj in gt:
            gt_rels.append(gt_rel.item())
            for pred_sub, pred_rel, pred_obj in pred:
                if pred_sub in human_index_map:
                    pred_sub = human_index_map[pred_sub]
                if pred_obj in human_index_map:
                    pred_obj = human_index_map[pred_obj]
                if gt_sub == pred_sub and gt_obj == pred_obj:
                    pred_rels.append(pred_rel)
                    break
            else:
                pred_rels.append(-1)
        accurary = float((torch.tensor(gt_rels) == torch.tensor(pred_rels)).sum()) / float(len(gt_rels))
        if accurary > max_acc:
            max_acc = accurary
            optimal_human_index_map = human_index_map
    return optimal_human_index_map


def get_rels_path(take_idx, USE_GT):
    if take_idx in TAKE_SPLIT['train']:
        if USE_GT:
            return Path('data/relationships_train.json')
        else:
            return Path('scan_relations_oracle_train.json')
    elif take_idx in TAKE_SPLIT['val']:
        if USE_GT:
            return Path('data/relationships_validation.json')
        else:
            return Path('scan_relations_oracle_val.json')
    elif take_idx in TAKE_SPLIT['test']:
        if USE_GT:
            return Path('data/relationships_test_dummy.json')
        else:
            return Path('scan_relations_oracle_test.json')

    return None


def get_take_rels(rels_path, take_idx, USE_GT):
    if USE_GT:
        with open(rels_path) as f:
            all_scans_rels = json.load(f)['scans']
            take_rels = [scan_gt_rels for scan_gt_rels in all_scans_rels if scan_gt_rels['take_idx'] == take_idx]
    else:
        if not rels_path.exists():
            return None
        with open(rels_path) as f:
            all_scans_rels = json.load(f)
            all_scans_rels = {k.rsplit('_', 1)[0]: v for k, v in all_scans_rels.items()}
            take_rels = []
            for key, value in all_scans_rels.items():
                t_idx, scan_idx = key.split('_')
                t_idx = int(t_idx)
                if t_idx == take_idx:
                    take_rels.append({'take_idx': t_idx, 'scan': scan_idx, 'relationships': value})

    return take_rels


def name_to_index(name):
    name_to_index = {
        'patient': 0,
        'head_surgeon': 1,
        'assistant_surgeon': 2,
        'circulating_nurse': 3,
        'anaesthetist': 4,
        'none': 5
    }
    return name_to_index[name]


def main():
    USE_GT_SCENE_GRAPHS = False
    print(f'USE GT : {USE_GT_SCENE_GRAPHS}')
    LABEL_NAMES = ['Patient', 'head_surgeon', 'assistant_surgeon', 'circulating_nurse', 'anaesthetist']
    split_to_all_gt_labels = defaultdict(list)
    split_to_all_pred_labels = defaultdict(list)

    # for take_idx in TAKE_SPLIT['train'] + TAKE_SPLIT['val'] + TAKE_SPLIT['test']:
    for take_idx in TAKE_SPLIT['train'] + TAKE_SPLIT['val']:
        all_gt_labels = []
        all_pred_labels = []
        rels_path = get_rels_path(take_idx, USE_GT_SCENE_GRAPHS)

        take_rels = get_take_rels(rels_path, take_idx, USE_GT_SCENE_GRAPHS)
        if take_rels is None:
            continue

        take_rels = sorted(take_rels, key=lambda x: x['scan'])
        if len(take_rels) == 0:
            continue

        # For evaluation, gt is still needed ofc
        gt_take_rels = get_take_rels(get_rels_path(take_idx, True, ), take_idx, True)
        for gt_sg in gt_take_rels:
            if len(gt_sg['relationships']) == 0:
                continue
            # 1) Find optimal matching over humans. 2) Check if this matches with the predicted roles.
            pred_sg = [sg for sg in take_rels if sg['scan'] == gt_sg['scan']]
            if len(pred_sg) == 0:
                continue
            pred_sg = pred_sg[0]
            human_roles = set()  # Need to be mapped for this evaluation
            if 'Patient' in gt_sg['objects'].values():
                gt_sg['human_idx_to_name']['patient'] = 'patient'
            is_patient_pred = False
            for sub, rel, obj in pred_sg['relationships']:
                if sub in ['head surgeon', 'assistant surgeon', 'circulator', 'nurse', 'anaesthetist']:
                    human_roles.add(sub)
                if obj in ['head surgeon', 'assistant surgeon', 'circulator', 'nurse', 'anaesthetist']:
                    human_roles.add(obj)
                if sub == 'patient' or obj == 'patient':
                    is_patient_pred = True
            human_roles_to_indices = {human_role: f'human_{idx}' for idx, human_role in enumerate(sorted(human_roles))}
            if is_patient_pred:
                human_roles_to_indices['patient'] = 'patient'
            reverse_human_roles_to_indices = {v: k for k, v in human_roles_to_indices.items()}
            pred_sg_without_roles = [(human_roles_to_indices.get(sub, sub).replace(' ', '_'), rel, human_roles_to_indices.get(obj, obj).replace(' ', '_')) for sub, rel, obj in
                                     pred_sg['relationships']]
            pred_sg_as_idx = []
            for sub, rel, obj in pred_sg_without_roles:
                try:
                    pred_sg_as_idx.append((map_scene_graph_name_to_vocab_idx(sub), map_scene_graph_name_to_vocab_idx(rel), map_scene_graph_name_to_vocab_idx(obj)))
                except Exception as e:
                    pass
            gt_sg_as_idx = [(gt_sg['objects'][str(sub_idx)], rel_name, gt_sg['objects'][str(obj_idx)]) for (sub_idx, obj_idx, rel_idx, rel_name) in gt_sg['relationships']]
            gt_sg_as_idx = [(map_scene_graph_name_to_vocab_idx(sub), map_scene_graph_name_to_vocab_idx(rel), map_scene_graph_name_to_vocab_idx(obj)) for (sub, rel, obj) in gt_sg_as_idx]
            optimal_human_indices = compute_optimal_human_indices(pred_sg_as_idx, torch.tensor(gt_sg_as_idx))

            for pred_index, gt_index in optimal_human_indices.items():
                # these are matching. Check there respective roles
                pred_human_index = map_vocab_idx_to_scene_graph_name(pred_index)
                gt_human_index = map_vocab_idx_to_scene_graph_name(gt_index)

                pred_role = reverse_human_roles_to_indices.get(pred_human_index, 'none').replace('circulator', 'circulating_nurse').replace('-', '_').replace(' ', '_')
                gt_role = gt_sg['human_idx_to_name'].get(gt_human_index, 'none').replace('-', '_')
                all_gt_labels.append(name_to_index(gt_role))
                all_pred_labels.append(name_to_index(pred_role))

        result = classification_report(all_gt_labels, all_pred_labels, labels=list(range(len(LABEL_NAMES))), target_names=LABEL_NAMES)
        print(f'TAKE {take_idx}')
        print(result)
        if take_idx in TAKE_SPLIT['train']:
            split_to_all_gt_labels['train'].extend(all_gt_labels)
            split_to_all_pred_labels['train'].extend(all_pred_labels)
        elif take_idx in TAKE_SPLIT['val']:
            split_to_all_gt_labels['val'].extend(all_gt_labels)
            split_to_all_pred_labels['val'].extend(all_pred_labels)
        else:
            split_to_all_gt_labels['test'].extend(all_gt_labels)
            split_to_all_pred_labels['test'].extend(all_pred_labels)

    train_results = classification_report(split_to_all_gt_labels['train'], split_to_all_pred_labels['train'], labels=list(range(len(LABEL_NAMES))),
                                          target_names=LABEL_NAMES)
    val_results = classification_report(split_to_all_gt_labels['val'], split_to_all_pred_labels['val'], labels=list(range(len(LABEL_NAMES))),
                                        target_names=LABEL_NAMES)
    test_results = classification_report(split_to_all_gt_labels['test'], split_to_all_pred_labels['test'], labels=list(range(len(LABEL_NAMES))),
                                         target_names=LABEL_NAMES)
    print(f'TRAIN')
    print(train_results)
    print(f'VAL')
    print(val_results)
    print(f'TEST')
    print(test_results)


if __name__ == '__main__':
    main()
