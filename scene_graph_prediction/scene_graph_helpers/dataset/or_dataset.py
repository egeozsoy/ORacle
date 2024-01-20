from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch
# import torch.utils.data as data
from torchvision import transforms as T

from helpers.configurations import OR_4D_DATA_ROOT_PATH
from helpers.utils import load_cam_infos
from scene_graph_prediction.scene_graph_helpers.dataset.augmentation_utils import apply_data_augmentation_to_object_pcs, \
    apply_data_augmentations_to_relation_pcs
from scene_graph_prediction.scene_graph_helpers.dataset.data_preparation_utils import data_preparation, load_full_image_data
from scene_graph_prediction.scene_graph_helpers.dataset.dataset_utils import load_data, get_weights, get_relationships, load_mesh, map_scene_graph_name_to_vocab_idx
from scene_graph_prediction.scene_graph_helpers.model.model_utils import get_image_model


class ORDataset(Dataset):
    def __init__(self,
                 config,
                 split='train',
                 shuffle_objs=False,
                 for_eval=False):

        assert split in ['train', 'val', 'test']
        self.split = split
        self.config = config
        self.mconfig = config['dataset']
        name_suffix = self.mconfig['DATASET_SUFFIX']
        if not self.config['USE_GT']:
            name_suffix += '_no_gt'
        if for_eval:
            name_suffix += '_eval'

        self.caching_folder = Path(f'{OR_4D_DATA_ROOT_PATH}/scene_graph_cache{name_suffix}')
        if not self.caching_folder.exists():
            self.caching_folder.mkdir()
        self.take_to_cam_infos = {}
        for take_idx in range(1, 11):
            self.take_to_cam_infos[take_idx] = load_cam_infos(OR_4D_DATA_ROOT_PATH / f'export_holistic_take{take_idx}_processed')

        self.root = self.mconfig['root']
        self.scans = []
        self.shuffle_objs = shuffle_objs
        self.sample_in_runtime = False
        self.for_eval = for_eval

        self.classNames, self.relationNames, self.data, self.selected_scans = load_data(self.root, self.config, self.mconfig, self.split, self.for_eval)
        self.w_cls_obj, self.w_cls_rel = get_weights(self.classNames, self.relationNames, self.data, self.selected_scans, for_eval=self.for_eval)

        self.relationship_json, self.objs_json, self.scans = get_relationships(self.data, self.selected_scans, self.classNames)

        assert (len(self.scans) > 0)

        self.cache_data = dict()
        self.take_idx_to_human_name_to_3D_joints = {}

        self.full_image_transformations = get_image_model(model_config=self.config['MODEL'], only_transforms=True)
        if self.full_image_transformations is not None:
            self.full_image_transformations = self.full_image_transformations[split]

            self.image_transform_pre = T.Compose(self.full_image_transformations.transforms[:2])
            self.image_transform_post = T.Compose(self.full_image_transformations.transforms[2:])

        if split == 'val' or split == 'test' and self.config['USE_VIS_DESC']:
            self.vis_knowledge_paths = [
            'data/original_crops/anesthesia equipment_take1.pt',
            'data/original_crops/cementing_take1.pt',
            'data/original_crops/cutting_take1.pt',
            'data/original_crops/drilling_take1.pt',
            'data/original_crops/hammering_take1.pt',
            'data/original_crops/sawing_take1.pt',
            'data/original_crops/suturing_take1.pt'
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
        scan_id = self.scans[index]
        scan_id_no_split = scan_id.rsplit('_', 1)[0]
        take_idx = scan_id.split('_')[0]
        if self.split != 'test':
            if take_idx in self.take_idx_to_human_name_to_3D_joints:
                human_name_to_3D_joints = self.take_idx_to_human_name_to_3D_joints[take_idx]
            else:
                human_name_to_3D_joints = np.load(str(OR_4D_DATA_ROOT_PATH / 'human_name_to_3D_joints' / f'{take_idx}_GT_True.npz'), allow_pickle=True)[
                    'arr_0'].item()
                self.take_idx_to_human_name_to_3D_joints[take_idx] = human_name_to_3D_joints
        else:
            human_name_to_3D_joints = None
        selected_instances = list(self.objs_json[scan_id].keys())
        map_instance2labelName = self.objs_json[scan_id]
        cache_path = self.caching_folder / f'{scan_id}.npz'
        image_input = self.config['IMAGE_INPUT']
        if cache_path.exists():
            sample = np.load(str(cache_path), allow_pickle=True)['arr_0'].item()
        else:
            sample = {'scan_id': scan_id, 'objs_json': self.objs_json[scan_id]}
            data = load_mesh(scan_id_no_split, scan_id, self.objs_json, self.config['USE_GT'], for_infer=self.for_eval,
                             human_name_to_3D_joints=human_name_to_3D_joints)
            points = data['points']
            instances = data['instances']
            instance_label_to_hand_locations = data['instance_label_to_hand_locations']
            obj_points, rel_points, edge_indices, instance2mask, relation_objects_one_hot, gt_rels, gt_class, rel_hand_points = \
                data_preparation(self.config, points, instances, selected_instances, self.mconfig['num_points_objects'],
                                 self.mconfig['num_points_relation'], for_train=True, instance2labelName=map_instance2labelName, classNames=self.classNames,
                                 rel_json=self.relationship_json[scan_id], relationships=self.relationNames, padding=0.2, shuffle_objs=self.shuffle_objs,
                                 instance_label_to_hand_locations=instance_label_to_hand_locations)

            sample['instance2mask'] = instance2mask
            sample['obj_points'] = obj_points
            sample['rel_points'] = rel_points
            sample['gt_class'] = gt_class
            sample['gt_rels'] = gt_rels
            sample['edge_indices'] = edge_indices
            sample['relation_objects_one_hot'] = relation_objects_one_hot
            sample['rel_hand_points'] = rel_hand_points

            np.savez_compressed(str(cache_path), sample)

        if self.split == 'train' and not self.for_eval and self.mconfig['data_augmentation']:
            p_value = 0.75
            if np.random.uniform(0, 1) < p_value:
                sample['obj_points'] = apply_data_augmentation_to_object_pcs(sample['obj_points'])
                sample['rel_points'] = apply_data_augmentations_to_relation_pcs(sample['rel_points'], sample['rel_hand_points'], sample['gt_rels'],
                                                                                self.relationNames)
        if image_input == 'full':
            sample['full_image'], sample['image_paths'] = load_full_image_data(scan_id_no_split, image_transform=self.full_image_transformations,
                                                                               augmentations=None)

        objs = self.objs_json[scan_id]
        relations = self.relationship_json[scan_id]
        # Map to our preferred form
        relations = [(objs[sub_idx], rel_name, objs[obj_idx]) for (sub_idx, obj_idx, rel_idx, rel_name) in relations]
        relations_tokenized = [(map_scene_graph_name_to_vocab_idx(sub), map_scene_graph_name_to_vocab_idx(rel), map_scene_graph_name_to_vocab_idx(obj)) for
                               (sub, rel, obj) in
                               relations]
        sample['relations_tokenized'] = relations_tokenized

        if self.config['USE_VIS_DESC']:
            if self.split == 'val' or self.split == 'test': #in validation and test we always need the same vis descriptors
                sample['vis_descriptor_embs'] = self.vis_descriptor_embs

        return sample
