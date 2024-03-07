import json

from helpers.configurations import TAKE_SPLIT


def main():
    # json_file_name = 'scan_relations_visual_only_val.json' # visual_only
    # json_file_name = 'scan_relations_visual_only_with_images_val.json' # visual_only_with_images
    # json_file_name = 'scan_relations_labrad-or_val.json' # labrad-or
    json_file_name = 'scan_relations_oracle_mv_learned_temporal_pred_val_mapped.json'  # labrad-or_with_images
    print(f'Calculating consistency for {json_file_name}...')
    with open(json_file_name) as f:
        data = json.load(f)
        macro_rel_consistency = []
        for take in TAKE_SPLIT['val']:
            take_rel_consistencies = []
            last_rels = None
            for scan_id in sorted([key for key in data.keys() if int(key.split('_')[0]) == take]):
                unique_rels = {pred for _, pred, _ in data[scan_id]}
                if last_rels is not None:
                    if len(unique_rels) == 0 and len(last_rels) == 0:
                        rel_consistency = 1
                    else:
                        rel_consistency = len(unique_rels.intersection(last_rels)) / len(unique_rels.union(last_rels))
                    if rel_consistency < 1:
                        print(f'{scan_id}: {unique_rels.symmetric_difference(last_rels)}')
                    take_rel_consistencies.append(rel_consistency)

                last_rels = unique_rels
            take_rel_consistency = sum(take_rel_consistencies) / len(take_rel_consistencies)
            print(f'Take {take} rel consistency: {take_rel_consistency:.4f}')
            macro_rel_consistency.append(take_rel_consistency)

        print(f'Macro rel consistency: {sum(macro_rel_consistency) / len(macro_rel_consistency):.4f}')


def main_gt():
    json_file_name = 'data/relationships_validation.json'
    print(f'Calculating consistency for GT {json_file_name}...')
    with open(json_file_name) as f:
        data = json.load(f)
        macro_rel_consistency = []
        for take in TAKE_SPLIT['val']:
            take_rel_consistencies = []
            last_rels = None
            for scan in sorted([scan for scan in data['scans'] if scan['take_idx'] == take], key=lambda x: x['scan']):
                unique_rels = {pred for _, _, _, pred in scan['relationships']}
                if last_rels is not None:
                    if len(unique_rels) == 0 and len(last_rels) == 0:
                        rel_consistency = 1
                    else:
                        rel_consistency = len(unique_rels.intersection(last_rels)) / len(unique_rels.union(last_rels))
                    take_rel_consistencies.append(rel_consistency)

                last_rels = unique_rels
            take_rel_consistency = sum(take_rel_consistencies) / len(take_rel_consistencies)
            print(f'Take {take} rel consistency: {take_rel_consistency:.4f}')
            macro_rel_consistency.append(take_rel_consistency)

        print(f'Macro rel consistency: {sum(macro_rel_consistency) / len(macro_rel_consistency):.4f}')


if __name__ == '__main__':
    main()
