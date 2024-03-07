import argparse

import json_tricks as json  # Allows to load integers etc. correctly
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torch

from torchvision import transforms as T

from helpers.configurations import MMOR_DATA_ROOT_PATH

from scene_graph_prediction.scene_graph_helpers.model.model_utils import get_image_model

scene_graph_name_to_vocab_idx = {
    'anesthesia_equipment': 0,
    'operating_table': 1,
    'instrument_table': 2,
    'secondary_table': 3,
    'instrument': 4,
    'patient': 6,
    'anaesthetist': 7,
    'nurse': 8,
    'mps': 9,
    'circulator': 10,
    'monitor': 15,
    'tracker': 16,
    'mps_station': 17,
    'mako_robot': 18,
    'unrelated_person': 19,
    'assisting': 20,
    'cementing': 21,
    'cleaning': 22,
    'closeto': 23,
    'cutting': 24,
    'drilling': 25,
    'hammering': 26,
    'holding': 27,
    'lyingon': 28,
    'operating': 29,
    'preparing': 30,
    'sawing': 31,
    'suturing': 32,
    'touching': 33
}
vocab_idx_to_scene_graph_name = {v: k for k, v in scene_graph_name_to_vocab_idx.items()}

synonyms = {
    'anesthesia_equipment': ['anaesthesia_equipment', 'anesthesia equipment', 'anaesthetist_station', 'ae'],
    'closeto': ['close', 'close to'],
    'operating': ['manipulating'],
    'instrument': ['tool'],
    'operating_table': ['opertating_table', 'ot'],
    'anaesthetist': ['anest']
}

role_synonyms = {
    'head_surgeon': ['head_surgent'],
    'anaesthetist': ['anesthetist'],
}


# Reverse synonym mapping
def reverse_synonym_mapping(synonyms_dict):
    reversed_dict = {}
    for key, synonyms_list in synonyms_dict.items():
        for synonym in synonyms_list:
            reversed_dict[synonym] = key
    return reversed_dict


# Applying the function
reversed_synonyms = reverse_synonym_mapping(synonyms)
reversed_role_synonyms = reverse_synonym_mapping(role_synonyms)


def map_scene_graph_name_to_vocab_idx(name):
    name = name.lower()
    # Synonym mapping
    if name in reversed_synonyms:
        name = reversed_synonyms[name]
    return scene_graph_name_to_vocab_idx[name]


def map_vocab_idx_to_scene_graph_name(vocab_idx):
    return vocab_idx_to_scene_graph_name[vocab_idx]


def load_full_image_data(scan_idx, image_transform, augmentations=None):
    with open(f'{MMOR_DATA_ROOT_PATH}/001_PKA/timestamp_to_pcd_and_frames_list.json') as f:
        timestamp_to_pcd_and_frames_list = json.load(f)
    images = []
    image_paths = []
    color_idx_str = timestamp_to_pcd_and_frames_list[int(scan_idx)][1]['azure']
    for c_idx in range(1, 6):
        color_path = Path(f'{MMOR_DATA_ROOT_PATH}/001_PKA/colorimage/camera0{c_idx}_colorimage-{color_idx_str}.jpg')
        image_paths.append(color_path)
        img = Image.open(color_path).convert('RGB')
        if augmentations is not None:
            img = augmentations(img)
        img = image_transform(img)
        images.append(img)

    return torch.stack(images), image_paths


class MMORDataset(Dataset):
    def __init__(self,
                 config,
                 mv_desc=False):

        self.config = config
        self.mconfig = config['dataset']
        name_suffix = self.mconfig['DATASET_SUFFIX']
        if not self.config['USE_GT']:
            name_suffix += '_no_gt'

        with open('data_mmor/relationships.json', "r") as read_file:
            self.data = json.load(read_file)

        self.scans = self.data['scans']

        self.full_image_transformations = get_image_model(model_config=self.config['MODEL'], only_transforms=True)
        if self.full_image_transformations is not None:
            self.full_image_transformations = self.full_image_transformations['val']

            self.image_transform_pre = T.Compose(self.full_image_transformations.transforms[:2])
            self.image_transform_post = T.Compose(self.full_image_transformations.transforms[2:])

        if self.config['USE_VIS_DESC']:
            if not mv_desc:
                self.vis_knowledge_paths = [
                    'data/original_crops/anesthesia equipment_take1.pt',
                    'data/original_crops/cementing_take1.pt',
                    'data/original_crops/cutting_take1.pt',
                    'data/original_crops/drilling_take1.pt',
                    'data/original_crops/hammering_take1.pt',
                    'data/original_crops/sawing_take1.pt',
                    'data/original_crops/suturing_take1.pt'
                ]

            else:
                self.vis_knowledge_paths = [
                    'data/original_crops/anesthesia equipment_cam2.pt',
                    'data/original_crops/anesthesia equipment_cam1.pt',
                    'data/original_crops/anesthesia equipment_cam3.pt',
                    'data/original_crops/anesthesia equipment_cam5.pt',
                    'data/original_crops/cementing_cam2.pt',
                    'data/original_crops/cementing_cam1.pt',
                    'data/original_crops/cementing_cam3.pt',
                    'data/original_crops/cementing_cam5.pt',
                    'data/original_crops/cutting_cam2.pt',
                    'data/original_crops/cutting_cam1.pt',
                    'data/original_crops/cutting_cam5.pt',
                    'data/original_crops/drilling_cam2.pt',
                    'data/original_crops/drilling_cam1.pt',
                    'data/original_crops/drilling_cam3.pt',
                    'data/original_crops/drilling_cam5.pt',
                    'data/original_crops/hammering_cam2.pt',
                    'data/original_crops/hammering_cam1.pt',
                    'data/original_crops/hammering_cam3.pt',
                    'data/original_crops/hammering_cam5.pt',
                    'data/original_crops/sawing_cam2.pt',
                    'data/original_crops/sawing_cam1.pt',
                    'data/original_crops/sawing_cam3.pt',
                    'data/original_crops/sawing_cam5.pt',
                    'data/original_crops/suturing_cam2.pt',
                    'data/original_crops/suturing_cam1.pt',
                    'data/original_crops/suturing_cam5.pt'
                ]

            self.vis_descriptor_embs = []
            for vis_knowledge_path in self.vis_knowledge_paths:
                vis_descriptor_emb = torch.load(vis_knowledge_path, map_location='cpu')
                self.vis_descriptor_embs.append(vis_descriptor_emb)

    def collate_fn(self, batch):
        for idx, elem in enumerate(batch):
            if 'obj_points' in elem:
                elem['obj_points'] = elem['obj_points'].permute(0, 2, 1)
            if 'rel_points' in elem:
                elem['rel_points'] = elem['rel_points'].permute(0, 2, 1)

            elem['gt_class'] = elem['gt_class'].flatten().long()
            elem['edge_indices'] = elem['edge_indices'].t().contiguous()
            elem['take_idx'] = int(elem['scan_id'].split('_')[0])

        return batch

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index):
        scan_id = self.scans[index]['scan']
        relations = self.scans[index]['relationships']
        image_input = self.config['IMAGE_INPUT']
        sample = {'scan_id': scan_id}

        if image_input == 'full':
            sample['full_image'], sample['image_paths'] = load_full_image_data(scan_id, image_transform=self.full_image_transformations,
                                                                               augmentations=None)

        # Map to our preferred form
        relations_tokenized = [(map_scene_graph_name_to_vocab_idx(sub), map_scene_graph_name_to_vocab_idx(rel), map_scene_graph_name_to_vocab_idx(obj)) for
                               (sub, rel, obj) in
                               relations]
        sample['relations_tokenized'] = relations_tokenized

        if self.config['USE_VIS_DESC']:
            sample['vis_descriptor_embs'] = self.vis_descriptor_embs

        return sample


def config_loader(config_path: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers/configs') / config_path
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint')
    args = parser.parse_args()
    config = config_loader(args.config)
    dataset = MMORDataset(config=config)
    for elem in dataset:
        a = 1
