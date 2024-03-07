from pathlib import Path
import json

from helpers.configurations import MMOR_DATA_ROOT_PATH


def filter_rels_by(scan_relationships, sub=None, obj=None, pred=None):
    filtered_rels = []
    for (s, o, p) in scan_relationships:
        if sub is not None and s != sub:
            continue
        if obj is not None and o != obj:
            continue
        if pred is not None and p != pred:
            continue
        filtered_rels.append((s, o, p))

    return filtered_rels


def infer_lyingon(scan_objects, scan_relationships):
    # if patient in scan_objects, infer patient lying on operating_table
    if 'patient' in scan_objects or 'ot' in scan_objects:
        scan_objects.add('patient')
        scan_objects.add('ot')
        scan_relationships.add((('patient', 'ot', 'LyingOn')))


def infer_operating_table_rels(scan_objects, scan_relationships):
    # if patient has any relations with an object, assume that object is close to the operating table
    new_rels = set()
    for s in scan_objects:
        if s == 'ot':
            continue

        rels1 = filter_rels_by(scan_relationships, sub=s, obj='patient')
        rels2 = filter_rels_by(scan_relationships, sub='patient', obj=s)
        if len(rels1) + len(rels2) > 0:
            existing_operating_table_rel = filter_rels_by(scan_relationships, sub=s, obj='ot')
            if len(existing_operating_table_rel) == 0:
                new_rels.add((s, 'ot', 'CloseTo'))

    scan_relationships.update(new_rels)


def infer_closeto(scan_objects, scan_relationships):
    # if any two objects have any relationship with each other, assume closeto in the other direction
    new_rels = set()
    for s1 in scan_objects:
        for s2 in scan_objects:
            rels1 = filter_rels_by(scan_relationships, sub=s1, obj=s2)
            rels2 = filter_rels_by(scan_relationships, sub=s2, obj=s1)
            if len(rels1) == 0 and len(rels2) == 0:
                # Both directions no relations, skip
                continue
            elif len(rels1) > 0 and len(rels2) > 0:
                # Both directions with relation, skip
                continue
            elif len(rels1) > 0 and len(rels2) == 0:
                # Relation only from s1 to s2, infer reverse closeTo relation
                new_rels.add((s2, s1, 'CloseTo'))
            elif len(rels1) == 0 and len(rels2) > 0:
                # Relation only from s2 to s1, infer reverse closeTo relation
                new_rels.add((s1, s2, 'CloseTo'))

    scan_relationships.update(new_rels)


def check_unique_relation(scan_objects, scan_relationships, relation_json_path):
    for s1 in scan_objects:
        for s2 in scan_objects:
            counter = 0
            for (sub, obj, pred) in scan_relationships:
                if (sub == s1 and obj == s2):
                    counter += 1

            assert counter <= 1
            # if counter > 1:
            #     print(f'Error: {relation_json_path.name} has multiple relations between {s1} and {s2}')


def main():
    save_path = Path('data_mmor')

    save_json_path = save_path / 'relationships.json'
    scans = []
    root_path = MMOR_DATA_ROOT_PATH / '001_PKA'
    relations_path = root_path / 'relation_labels'

    for relation_json_path in sorted(list(relations_path.glob('*.json'))):
        frame_id = relation_json_path.name.replace('.json', '')
        scan_objects = set()
        scan_relationships = set()

        with relation_json_path.open() as f:
            info_json = json.load(f)
            relation_json = info_json['rel_annotations']

        for sub, pred, obj in relation_json:
            scan_objects.add(sub)
            scan_objects.add(obj)
            scan_relationships.add((sub, obj, pred))

        infer_lyingon(scan_objects, scan_relationships)
        infer_operating_table_rels(scan_objects, scan_relationships)
        infer_closeto(scan_objects, scan_relationships)
        check_unique_relation(scan_objects, scan_relationships, relation_json_path)

        scans.append(
            {'scan': frame_id, 'relationships': list(scan_relationships)})

    relationship_json = {'scans': []}

    for scan in scans:
        relationship_json['scans'].append(scan)

    with save_json_path.open('w') as f:
        json.dump(relationship_json, f)


if __name__ == '__main__':
    main()
