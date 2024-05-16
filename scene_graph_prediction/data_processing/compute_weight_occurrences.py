import numpy as np


def compute_weights(labels, classes, count, verbose=False):
    if verbose: print("-------------")
    sum_weights = 0
    for c in range(len(classes)):
        if classes[c] / count > 0:
            sum_weights += count / classes[c]

    sum_weight_norm = 0
    weights = list()
    for c in range(len(classes)):
        if classes[c] / count > 0:
            weight = count / classes[c] / sum_weights
            sum_weight_norm += weight
            if verbose: print('{0:>20s} {1:>1.3f} {2:>6d}'.format(labels[c], weight, int(classes[c])))
            weights.append(weight)
        else:
            if verbose: print('{0:>20s} {1:>1.3f} {2:>6d}'.format(labels[c], 0.0, int(classes[c])))
            weights.append(0)
    if verbose: print("-------------")
    return weights


def compute(classNames, relationNames, relationship_data, selections: list = None, verbose=False):
    o_rel_cls = np.zeros((len(relationNames)))
    o_obj_cls = np.zeros((len(classNames)))
    classes_count = 0
    counter = 0

    exceed_ids = dict()
    scene_analysis = dict()
    cnn = 0
    for scan in relationship_data['scans']:
        scan_id = f'{scan["take_idx"]}_{scan["scan"]}'
        if selections is not None:
            if scan_id not in selections:
                continue
        instance2LabelName = {}

        for k, v in scan["objects"].items():
            instance2LabelName[int(k)] = v
            if v not in classNames:
                if verbose: print(v, 'not in classNames')
            o_obj_cls[classNames.index(v)] += 1

        nnk = dict()
        for relationship in scan["relationships"]:
            if relationship[3] not in relationNames:
                if verbose: print(relationship[3], 'not in relationNames')
                continue

            obj = relationship[0]  # id object
            sub = relationship[1]  # id subject
            rel = relationship[2]  # id relationship

            if obj == 0 or sub == 0:
                raise RuntimeError('found obj or sub is 0')

            if not obj in instance2LabelName:
                RuntimeWarning('key not found:', obj)
                continue

            if not sub in instance2LabelName:
                RuntimeWarning('key not found:', sub)
                continue

            if relationNames.index(relationship[3]) >= len(relationNames):
                if rel not in exceed_ids:
                    exceed_ids[relationNames.index(relationship[3])] = 0
                else:
                    exceed_ids[relationNames.index(relationship[3])] += 1
                continue
            o_rel_cls[relationNames.index(relationship[3])] += 1
            classes_count += 1

            nn = str(obj) + '_' + str(sub)
            if nn not in nnk:
                nnk[nn] = 0
            nnk[str(obj) + '_' + str(sub)] += 1
        for v in nnk.values():
            if v > 1:
                cnn += 1

        counter += 1

        scene_analysis[scan_id] = dict()
        scene_analysis[scan_id]['num objects'] = len(scan["objects"].items())
        scene_analysis[scan_id]['num relationships'] = len(scan['relationships'])
    if verbose: print('num multi predicates:', cnn)

    if len(exceed_ids) > 1:
        print('exceed map')
        for id, count in exceed_ids.items():
            print('\t', id, count)

    if verbose: print("objects:")
    wobjs = compute_weights(classNames, o_obj_cls, classes_count, verbose)
    if verbose: print("relationships:")
    wrels = compute_weights(relationNames, o_rel_cls, classes_count, verbose)
    return wobjs, wrels, o_obj_cls, o_rel_cls
