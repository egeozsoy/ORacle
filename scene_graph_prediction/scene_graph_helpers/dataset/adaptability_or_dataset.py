import json
from pathlib import Path

from torch.utils.data import Dataset


class AdaptabilityORDataset(Dataset):
    def __init__(self,
                 config,
                 root_path='adaptability_4dor'):

        self.config = config
        self.root_path = Path(root_path)
        self.labels = []

        self.all_test_cases = []

        for folder in sorted(self.root_path.iterdir()):
            if folder.is_dir():
                self.labels.append(folder.name)

        for label in self.labels:
            for json_file_path in (self.root_path / label / 'positives').glob('*.json'):
                positive_image_path = json_file_path.with_suffix('.jpg').absolute()
                negative_image_path = (positive_image_path.parent.parent / 'negatives' / positive_image_path.name).absolute()
                with json_file_path.open('r') as f:
                    json_data = json.load(f)
                self.all_test_cases.append((json_data, positive_image_path, label, True))
                self.all_test_cases.append((json_data, negative_image_path, label, False))

    def __len__(self):
        return len(self.all_test_cases)

    def __getitem__(self, index):
        json_data, image_path, label, is_pos = self.all_test_cases[index]
        vis_desc_path = (image_path.parent.parent / 'visual_features') / json_data['vis_img_name']
        test_condition = json_data['test_condition']
        textual_attributes = json_data['textual_attributes']
        return {'image_path': image_path, 'vis_desc_path': vis_desc_path, 'test_condition': test_condition, 'textual_attributes': textual_attributes, 'label': label, 'is_pos': is_pos}
