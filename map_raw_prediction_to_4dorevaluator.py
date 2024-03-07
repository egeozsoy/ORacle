import json
from pathlib import Path

from scene_graph_prediction.scene_graph_helpers.dataset.dataset_utils import reversed_role_synonyms, map_scene_graph_name_to_vocab_idx, map_vocab_idx_to_scene_graph_name

scene_graph_name_to_vocab_idx = {
    'anesthesia_equipment': 0,
    'operating_table': 1,
    'instrument_table': 2,
    'secondary_table': 3,
    'instrument': 4,
    'object': 5,
    'patient': 6,
    'human_0': 7,
    'human_1': 8,
    'human_2': 9,
    'human_3': 10,
    'human_4': 11,
    'Assisting': 20,
    'Cementing': 21,
    'Cleaning': 22,
    'CloseTo': 23,
    'Cutting': 24,
    'Drilling': 25,
    'Hammering': 26,
    'Holding': 27,
    'LyingOn': 28,
    'Operating': 29,
    'Preparing': 30,
    'Sawing': 31,
    'Suturing': 32,
    'Touching': 33
}
vocab_idx_to_scene_graph_name = {v: k for k, v in scene_graph_name_to_vocab_idx.items()}


def map_triplets_to_4dor_eval(triplets):
    human_roles = set()  # Need to be mapped for this evaluation
    tmp_mapped_triplets = []
    for (sub, pred, obj) in triplets:
        if sub in reversed_role_synonyms:
            sub = reversed_role_synonyms[sub]
        if obj in reversed_role_synonyms:
            obj = reversed_role_synonyms[obj]
        if sub in ['head surgeon', 'assistant surgeon', 'circulator', 'nurse', 'anaesthetist', 'patient']:
            human_roles.add(sub)
        if obj in ['head surgeon', 'assistant surgeon', 'circulator', 'nurse', 'anaesthetist', 'patient']:
            human_roles.add(obj)

        tmp_mapped_triplets.append((sub, pred, obj))

    human_roles_to_indices = {human_role: f'human_{idx}' for idx, human_role in enumerate(sorted(human_roles))}

    mapped_triplets = []

    for sub, pred, obj in tmp_mapped_triplets:
        try:
            sub = vocab_idx_to_scene_graph_name[map_scene_graph_name_to_vocab_idx(human_roles_to_indices.get(sub, sub).replace(' ', '_'))]
            obj = vocab_idx_to_scene_graph_name[map_scene_graph_name_to_vocab_idx(human_roles_to_indices.get(obj, obj).replace(' ', '_'))]
            pred = vocab_idx_to_scene_graph_name[map_scene_graph_name_to_vocab_idx(pred)]
            if sub == obj:
                continue
            mapped_triplets.append((sub, pred, obj))
        except Exception as e:
            print(f'Error: {e}')
            continue

    return mapped_triplets


def main():
    raw_triplets_json_path = Path('scan_relations_oracle_vis_desc_mv_test_22500_occaugs_visdesc_nosawing.json')
    with raw_triplets_json_path.open('r') as f:
        raw_triplets: dict = json.load(f)  # scan_id to triplets

    mapped_triplets: dict = {scan_id: map_triplets_to_4dor_eval(triplets) for scan_id, triplets in raw_triplets.items()}

    mapped_name = raw_triplets_json_path.stem + '_mapped.json'
    with (raw_triplets_json_path.parent / mapped_name).open('w') as f:
        json.dump(mapped_triplets, f)


if __name__ == '__main__':
    main()
