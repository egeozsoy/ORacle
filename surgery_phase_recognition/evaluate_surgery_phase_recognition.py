# Calculate True label per frame and predicted label. At the very end, this is simply a multiclass classification problem, which can be solved with sklearn.
# We want a score both per take, and for entire dataset.
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report

from helpers.configurations import TAKE_SPLIT
from surgery_phase_recognition.config import PHASE_ORDER, PHASE_LONG_NAMES

if __name__ == '__main__':
    root_path = Path('surgery_phase_recognition/phases_to_frames')

    for split_name in ['test']:
        all_gts = []
        all_preds = []
        for take_idx in TAKE_SPLIT[split_name]:
            with (root_path / f'phase_to_frames_{take_idx}.json').open('r') as f:
                gt = json.load(f)
            with (root_path / f'scan_relations_oracle_mv_learned_temporal_pred_{split_name.replace("val", "validation")}_phase_to_frames_{take_idx}.json').open('r') as f:  # TODO adjust as necessary
                pred = json.load(f)

            # Both are in form of phase: (start, end). What we instead want is an array of length video, where each element is the phase.
            take_length = np.asarray(list(gt.values())).max() + 1
            gts = np.zeros(take_length, dtype=int) - 1
            preds = np.zeros(take_length, dtype=int) - 1

            for phase, (start, end) in gt.items():
                gts[start:end + 1] = PHASE_ORDER.index(phase)
            for phase, (start, end) in pred.items():
                preds[start:end + 1] = PHASE_ORDER.index(phase)

            gts = list(gts)
            preds = list(preds)
            all_gts.extend(gts)
            all_preds.extend(preds)

            cls_report = classification_report(gts, preds, labels=list(range(len(PHASE_ORDER))),
                                               target_names=PHASE_LONG_NAMES, output_dict=False)

            print(f'\nTake {take_idx}\n')
            print(cls_report)

        cls_report = classification_report(all_gts, all_preds, labels=list(range(len(PHASE_ORDER))),
                                           target_names=PHASE_LONG_NAMES, output_dict=False)

        print(f'\n{split_name}\n')
        print(cls_report)
