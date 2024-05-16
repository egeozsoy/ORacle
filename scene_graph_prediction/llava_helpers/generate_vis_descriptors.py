# 1) Define Entities to Add. Define corresponding predicates.
# 2) Define logic for taking an existing scene from 4D-OR and adding the new entities. and corresponding predicates. (update the scene graph) and the descriptors.
# 3) Save new scene image, new scene graph, and new descriptors. (new data point)

import json
import random
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from helpers.configurations import TAKE_SPLIT, OR_4D_DATA_ROOT_PATH

replacement_map = {
    'Surgical Drill': {'replace_pred_from': ['Cleaning', 'Cutting', 'Drilling', 'Suturing', 'Touching'], 'replace_pred_to': 'Drilling'},
    'Surgical Bone Saw': {'replace_pred_from': ['Cleaning', 'Cutting', 'Sawing', 'Suturing', 'Touching'], 'replace_pred_to': 'Sawing'},
    'Surgical Hammer': {'replace_pred_from': ['Cutting', 'Hammering', 'Suturing', 'Touching'], 'replace_pred_to': 'Hammering'},
    'Surgical Scissors': {'replace_pred_from': ['Cleaning', 'Cutting', 'Suturing', 'Touching'], 'replace_pred_to': 'Suturing'},
    'Surgical Retractor': {'replace_pred_from': ['Assisting', 'Cleaning', 'Touching'], 'replace_pred_to': 'Retracting'},
    'Surgical Scalpel': {'replace_pred_from': ['Cleaning', 'Cutting', 'Suturing', 'Touching'], 'replace_pred_to': 'Cutting'},
    'Surgical Forceps': {'replace_pred_from': ['Assisting', 'Cleaning', 'Touching'], 'replace_pred_to': 'Grasping'},
    'Surgical Bone Cement Gun': {'replace_pred_from': ['Cementing', 'Cleaning', 'Cutting', 'Suturing', 'Touching'], 'replace_pred_to': 'Cementing'},

    'da Vinci Surgical System': {'replace_entity_from': ['anesthesia_equipment', 'instrument_table', 'secondary_table'],
                                 'replace_entity_to': 'da Vinci Surgical System'},
    'Mako Robotic-Arm Assisted Surgery System': {'replace_entity_from': ['anesthesia_equipment', 'instrument_table', 'secondary_table'],
                                                 'replace_entity_to': 'Mako Robotic-Arm Assisted Surgery System'},
    'Electrosurgical Unit': {'replace_entity_from': ['anesthesia_equipment', 'instrument_table', 'secondary_table'], 'replace_entity_to': 'Electrosurgical Unit'},
    'Surgical C-arm': {'replace_entity_from': ['anesthesia_equipment', 'instrument_table', 'secondary_table'], 'replace_entity_to': 'Surgical C-arm'},
    'Microscope for Microsurgery': {'replace_entity_from': ['anesthesia_equipment', 'instrument_table', 'secondary_table'],
                                    'replace_entity_to': 'Microscope for Microsurgery'},
    'Anesthesia Equipment': {'replace_entity_from': ['anesthesia_equipment', 'instrument_table', 'secondary_table'], 'replace_entity_to': 'anesthesia_equipment'},
    'Surgical Navigation System': {'replace_entity_from': ['anesthesia_equipment', 'instrument_table', 'secondary_table'],
                                   'replace_entity_to': 'Surgical Navigation System'},
    'Surgical Imaging Systems': {'replace_entity_from': ['anesthesia_equipment', 'instrument_table', 'secondary_table'],
                                 'replace_entity_to': 'Surgical Imaging Systems'}
}

# we have to define a map of mutually exclusive replacement.
INSTRUMENTS = {'Surgical Drill', 'Surgical Bone Saw', 'Surgical Hammer', 'Surgical Scissors', 'Surgical Retractor', 'Surgical Scalpel', 'Surgical Forceps', 'Surgical Bone Cement Gun'}
EQUIPMENT = {'da Vinci Surgical System', 'Mako Robotic-Arm Assisted Surgery System', 'Electrosurgical Unit', 'Surgical C-arm', 'Microscope for Microsurgery', 'Anesthesia Equipment',
             'Surgical Navigation System', 'Surgical Imaging Systems'}

# all to lower case
replacement_map = {k.lower(): v for k, v in replacement_map.items()}
INSTRUMENTS = {elem.lower() for elem in INSTRUMENTS}
EQUIPMENT = {elem.lower() for elem in EQUIPMENT}


def _load_gt_scene_graphs_in_prediction_format():
    all_scan_relations = {}
    for take_idx in TAKE_SPLIT['train']:
        if take_idx in TAKE_SPLIT['train']:
            gt_rels_path = Path('scene_graph_data/relationships_train.json')
        elif take_idx in TAKE_SPLIT['val']:
            gt_rels_path = Path('scene_graph_data/relationships_validation.json')
        elif take_idx in TAKE_SPLIT['test']:
            gt_rels_path = Path('scene_graph_data/relationships_test.json')
        else:
            raise Exception()
        with open(gt_rels_path) as f:
            all_scans_gt_rels = json.load(f)['scans']
        take_gt_rels = [scan_gt_rels for scan_gt_rels in all_scans_gt_rels if scan_gt_rels['take_idx'] == take_idx]
        take_gt_rels = sorted(take_gt_rels, key=lambda x: x['scan'])
        if len(take_gt_rels) == 0:
            continue
        for scan_gt_rels in take_gt_rels:
            object_idx_to_name = scan_gt_rels['objects']
            rels = []
            for sub_idx, obj_idx, rel_idx, rel_name in scan_gt_rels['relationships']:
                rels.append((object_idx_to_name[str(sub_idx)], rel_name, object_idx_to_name[str(obj_idx)]))
            all_scan_relations[f'{scan_gt_rels["take_idx"]}_{scan_gt_rels["scan"]}'] = rels

    return all_scan_relations


def _load_gt_role_labels(take_indices):
    take_frame_to_human_idx_to_name_and_joints = {}
    for take_idx in take_indices:
        root_path = OR_4D_DATA_ROOT_PATH / 'human_name_to_3D_joints'
        GT_take_human_name_to_3D_joints = np.load(str(root_path / f'{take_idx}_GT_True.npz'), allow_pickle=True)['arr_0'].item()
        if take_idx in TAKE_SPLIT['train']:
            gt_rels_path = Path('scene_graph_data/relationships_train.json')
        elif take_idx in TAKE_SPLIT['val']:
            gt_rels_path = Path('scene_graph_data/relationships_validation.json')
        elif take_idx in TAKE_SPLIT['test']:
            gt_rels_path = Path('scene_graph_data/relationships_test.json')
        else:
            raise Exception()
        with open(gt_rels_path) as f:
            all_scans_gt_rels = json.load(f)['scans']
        for scan_gt_rel in all_scans_gt_rels:
            if scan_gt_rel['take_idx'] != take_idx:
                continue
            if 'Patient' in scan_gt_rel['objects'].values():
                scan_gt_rel['human_idx_to_name']['Patient'] = 'Patient'
            take_frame_str = f'{take_idx}_{scan_gt_rel["scan"]}'
            human_indices = list(scan_gt_rel['human_idx_to_name'].keys())
            human_idx_to_human_name_and_joints = {}
            for human_idx in human_indices:
                try:
                    name = scan_gt_rel['human_idx_to_name'][human_idx]
                    joints = GT_take_human_name_to_3D_joints[scan_gt_rel["scan"]][human_idx]
                    human_idx_to_human_name_and_joints[human_idx] = (name, joints)
                except Exception as e:
                    continue

            take_frame_to_human_idx_to_name_and_joints[take_frame_str] = human_idx_to_human_name_and_joints

    return take_frame_to_human_idx_to_name_and_joints


def _preprocess_metadata(metadata, entity_crops_path: Path, view_indices):
    if 'none' in metadata: metadata.pop('none')
    all_entities = set(metadata.keys())
    entity_to_valid_names = {}
    for view_idx in view_indices:
        camera_entity_crops_path = entity_crops_path / str(view_idx)
        for entity in all_entities:
            entity_path = camera_entity_crops_path / entity
            if not entity_path.exists():
                print(f'Entity {entity} does not exist in view {view_idx}')
                continue
            if entity not in entity_to_valid_names:
                entity_to_valid_names[entity] = {elem.name.replace('.png', '') for elem in entity_path.glob('*.png')}
            else:
                entity_to_valid_names[entity] &= {elem.name.replace('.png', '') for elem in entity_path.glob('*.png')}
    for key, values in metadata.items():
        indices_to_remove = []
        for value_idx, value in enumerate(values):
            if value['image_name'] not in entity_to_valid_names[key] or value['c_idx'] not in view_indices:
                indices_to_remove.append(value_idx)
            metadata[key][value_idx]['entity_triplets'] = sorted([tuple(v) for v in value['entity_triplets']])
        metadata[key] = [v for i, v in enumerate(values) if i not in indices_to_remove]


def _sample_suitable_scene(graphs, entities_needed, predicates_needed):
    suitable_scenes = []
    for idx, graph in graphs.items():
        entities = {elem[0] for elem in graph} | {elem[2] for elem in graph}
        predicates = {elem[1] for elem in graph if elem[2] == 'Patient'}
        # remove any predicate that is used multiple times in the scene
        predicates = {elem for elem in predicates if len([elem2 for elem2 in graph if elem2[1] == elem]) == 1}

        # if at least one entity from entities_needed is in entities and at least one predicate from predicates_needed is in predicates then add to suitable_scenes
        if (len(entities_needed & entities) > 0 or len(entities_needed) == 0) and (len(predicates_needed & predicates) > 0 or len(predicates_needed) == 0):
            suitable_scenes.append({'suitable_entities': entities_needed & entities, 'suitable_predicates': predicates_needed & predicates, 'idx': idx})
    return random.choice(suitable_scenes)


def _sample_object_with_descriptor(object_type_to_images, is_instrument, is_equipment):
    if is_instrument:
        object_type = np.random.choice(list(INSTRUMENTS)).lower()
    elif is_equipment:
        object_type = np.random.choice(list(EQUIPMENT)).lower()
    else:
        raise ValueError('Only one of is_instrument and is_equipment can be true')
    image, attrs = random.choice(object_type_to_images[object_type])
    with open(attrs) as f:
        attrs = json.load(f)
    return image, attrs


def init_worker():
    global object_type_to_images, export_path, take_to_bg, meta_data, entity_crops_path, graphs, role_labels, VIEW_INDICES, OCC_AUGS, NUM_OCC_AUGS, replacement_map, list_of_syn_objects, take_to_view_to_bg  # Efficient, as each worker will have its own copy of these variables, no need to share/serialize them.
    VIEW_INDICES = (2,)  # (2, 1, 3, 5)
    OCC_AUGS = False
    NUM_OCC_AUGS = 10

    export_path = Path('synthetic_or_generation/vis_descriptors')
    export_path.mkdir(exist_ok=True, parents=True)

    # 1. First fetch the background images
    take_to_view_to_bg = defaultdict(dict)
    for take_idx in [1, 3, 5, 7, 9, 10]:
        for view_idx in VIEW_INDICES:
            bg = Image.open(f'synthetic_or_generation/clean_bgs/take_{take_idx}_cam_{view_idx}_clean_bg.jpg').convert('RGBA')
            take_to_view_to_bg[take_idx][view_idx] = bg

    # 2. Fetch the metadata, and preprocess it
    entity_crops_path = Path('synthetic_or_generation/entity_crops_all')
    with open(entity_crops_path / 'metadata.json') as f:
        meta_data = json.load(f)
    _preprocess_metadata(meta_data, entity_crops_path, VIEW_INDICES)

    graphs = _load_gt_scene_graphs_in_prediction_format()
    role_labels = _load_gt_role_labels(TAKE_SPLIT['train'] + TAKE_SPLIT['val'] + TAKE_SPLIT['test'])


def main_worker(d_idx):
    try:
        object_type, object_img, object_attr_path = list_of_syn_objects[d_idx]
        with open(object_attr_path) as f:
            object_attrs = json.load(f)
        instrument_image = object_img if object_type in INSTRUMENTS else None
        equipment_image = object_img if object_type in EQUIPMENT else None

        # Find the minimum requirements that a 4D-OR scene should have to fullfill the conditations.
        entities_needed = set()  # one of these should be in the scene
        predicates_needed = set()  # one of these should be in the scene
        if instrument_image:
            predicates_needed.update(replacement_map[object_attrs['object_type']]['replace_pred_from'])
        if equipment_image:
            entities_needed.update(replacement_map[object_attrs['object_type']]['replace_entity_from'])

        found_suitable_scene = False
        while not found_suitable_scene:
            ref_scene = _sample_suitable_scene(graphs, entities_needed, predicates_needed)
            if equipment_image:
                # find the entity with that pred
                entity_to_replace = random.choice(list(ref_scene['suitable_entities']))
                for view_idx in VIEW_INDICES:  # check if the entity is visible from at least 1 view
                    entity_metadata = [elem for elem in meta_data[entity_to_replace] if elem['image_name'] == ref_scene['idx'] and elem['c_idx'] == view_idx]
                    if len(entity_metadata) > 0:
                        found_suitable_scene = True
                        break
            else:
                found_suitable_scene = True  # entity_to_replace will be selected later and should be always valid

        ref_take, ref_idx = ref_scene['idx'].split('_')
        ref_graph = graphs[ref_scene['idx']]
        roles = role_labels[ref_scene['idx']]
        data = {'sg': ref_graph, 'descriptors': {}, 'replaced_pred': None, 'replaced_entity': None, 'paths': {}}
        view_to_rgb = {}
        view_to_rgb_augs = defaultdict(list)

        if instrument_image:
            # locate the left or right hand of the surgeon. Place the instrument image there.
            pred_to_replace = random.choice(list(ref_scene['suitable_predicates']))
            # find the entity with that pred
            staff_with_pred = random.choice([elem[0] for elem in ref_graph if elem[1] in pred_to_replace])
            staff_with_pred = roles[staff_with_pred][0]
            if 'head-surgeon' not in staff_with_pred.lower():
                return (False, d_idx)
            # find left or right hand of that entity
            left_or_right = random.choice(['left', 'right'])
            root_path = OR_4D_DATA_ROOT_PATH / f'export_holistic_take{ref_take}_processed'
            color_image_path = root_path / 'colorimage'
            with (root_path / 'timestamp_to_pcd_and_frames_list.json').open() as f:
                timestamp_to_pcd_and_frames_list = json.load(f)

            instrument_image_path = instrument_image.stem
            instrument_image = Image.open(instrument_image).convert('RGBA')
            instrument_image_paths = []
            for view_idx in VIEW_INDICES:
                color_image_str = timestamp_to_pcd_and_frames_list[int(ref_idx)][1][f'color_{view_idx}']
                rgb_path = color_image_path / f'camera0{view_idx}_colorimage-{color_image_str}.jpg'
                if view_idx not in view_to_rgb:
                    rgb = Image.open(str(rgb_path)).convert("RGBA")
                    view_to_rgb[view_idx] = rgb
                else:
                    rgb = view_to_rgb[view_idx]
                entity_metadata = [elem for elem in meta_data[staff_with_pred] if elem['image_name'] == ref_scene['idx'] and elem['c_idx'] == view_idx]
                if len(entity_metadata) == 0:
                    view_to_rgb[view_idx] = None
                    continue
                entity_metadata = entity_metadata[0]
                entity_depth = entity_metadata['average_depth']
                # sample the entity image from that corresponding frame to be used as mask on top of the instrument image
                hand_position = entity_metadata['left_hand'] if left_or_right == 'left' else entity_metadata['right_hand']
                if hand_position is None:
                    hand_position = entity_metadata['xmax'] - entity_metadata['xmin'], entity_metadata['ymax'] - entity_metadata['ymin']  # default to entity center

                instrument_image_paths.append(instrument_image_path)

                if OCC_AUGS:
                    for _ in range(NUM_OCC_AUGS):
                        curr_rgb = deepcopy(rgb)
                        # scale the object depending on the depth of the entity. Meaning the further away the entity is, the smaller the object should be.
                        scale_factor = random.randint(200_000, 400_000)
                        object_size = scale_factor / entity_depth
                        curr_instrument_image = deepcopy(instrument_image)
                        curr_instrument_image = curr_instrument_image.resize((int(object_size), int(object_size)))
                        # cut away pieces of the object
                        curr_instrument_image = transforms.RandomResizedCrop(curr_instrument_image.size, scale=(0.6, 1.0), ratio=(0.85, 1.15))(curr_instrument_image)
                        # plot so that center of instrument image is at hand_position
                        curr_rgb.paste(curr_instrument_image, (hand_position[0] - curr_instrument_image.width // 2, hand_position[1] - curr_instrument_image.height // 2), curr_instrument_image)

                        # crop around the instrument
                        margin_size = 30  # adjust this value as needed

                        # Calculate the top-left corner of the bounding box with margin
                        top_left_x = max(hand_position[0] - curr_instrument_image.width // 2 - margin_size, 0)
                        top_left_y = max(hand_position[1] - curr_instrument_image.height // 2 - margin_size, 0)

                        # Calculate the bottom-right corner of the bounding box with margin
                        bottom_right_x = min(top_left_x + curr_instrument_image.width + 2 * margin_size, curr_rgb.width)
                        bottom_right_y = min(top_left_y + curr_instrument_image.height + 2 * margin_size, curr_rgb.height)

                        # Define the bounding box for cropping with margin
                        bounding_box = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

                        # Crop the image with margin
                        curr_rgb = curr_rgb.crop(bounding_box)
                        view_to_rgb_augs[view_idx].append(curr_rgb)

                else:
                    curr_rgb = deepcopy(rgb)
                    # scale the object depending on the depth of the entity. Meaning the further away the entity is, the smaller the object should be.
                    scale_factor = random.randint(200_000, 400_000)
                    object_size = scale_factor / entity_depth
                    instrument_image = instrument_image.resize((int(object_size), int(object_size)))

                    # plot so that center of instrument image is at hand_position
                    curr_rgb.paste(instrument_image, (hand_position[0] - instrument_image.width // 2, hand_position[1] - instrument_image.height // 2),
                                   instrument_image)

                    # crop around the instrument
                    margin_size = 30  # adjust this value as needed

                    # Calculate the top-left corner of the bounding box with margin
                    top_left_x = max(hand_position[0] - instrument_image.width // 2 - margin_size, 0)
                    top_left_y = max(hand_position[1] - instrument_image.height // 2 - margin_size, 0)

                    # Calculate the bottom-right corner of the bounding box with margin
                    bottom_right_x = min(top_left_x + instrument_image.width + 2 * margin_size, rgb.width)
                    bottom_right_y = min(top_left_y + instrument_image.height + 2 * margin_size, rgb.height)

                    # Define the bounding box for cropping with margin
                    bounding_box = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

                    # Crop the image with margin
                    curr_rgb = curr_rgb.crop(bounding_box)
                    view_to_rgb[view_idx] = curr_rgb

            # now correctly update the scene graph
            new_graph = []
            for sub, rel, obj in ref_graph:
                if rel == pred_to_replace:
                    rel = replacement_map[object_attrs['object_type']]['replace_pred_to']
                new_graph.append((sub, rel, obj))
            ref_graph = new_graph

            data['descriptors'][replacement_map[object_attrs['object_type']]['replace_pred_to']] = object_attrs
            data['paths'][replacement_map[object_attrs['object_type']]['replace_pred_to']] = instrument_image_paths
            data['replaced_pred'] = pred_to_replace

        if equipment_image:
            # sample the entity image from that corresponding frame to be used as mask on top of the instrument image
            root_path = OR_4D_DATA_ROOT_PATH / f'export_holistic_take{ref_take}_processed'
            color_image_path = root_path / 'colorimage'
            with (root_path / 'timestamp_to_pcd_and_frames_list.json').open() as f:
                timestamp_to_pcd_and_frames_list = json.load(f)

            equipment_image_path = equipment_image.stem
            equipment_image = Image.open(equipment_image).convert('RGBA')
            equipment_image_paths = []
            for view_idx in VIEW_INDICES:
                color_image_str = timestamp_to_pcd_and_frames_list[int(ref_idx)][1][f'color_{view_idx}']
                rgb_path = color_image_path / f'camera0{view_idx}_colorimage-{color_image_str}.jpg'
                if view_idx not in view_to_rgb:
                    rgb = Image.open(str(rgb_path)).convert("RGBA")
                    view_to_rgb[view_idx] = rgb
                else:
                    rgb = view_to_rgb[view_idx]
                entity_metadata = [elem for elem in meta_data[entity_to_replace] if elem['image_name'] == ref_scene['idx'] and elem['c_idx'] == view_idx]
                if len(entity_metadata) == 0:
                    view_to_rgb[view_idx] = None
                    continue
                entity_metadata = entity_metadata[0]
                entity_depth = entity_metadata['average_depth'] if entity_metadata['average_depth'] is not None else 5000
                # 1) Clean where the current entity is. Use clean bg for this.
                entity_mask = Image.open(entity_crops_path / f'{view_idx}/{entity_to_replace}/{ref_scene["idx"]}.png')
                # Mask cleanbg using entity mask
                clean_bg = deepcopy(take_to_view_to_bg[int(ref_take)][view_idx]).convert('RGBA')
                clean_bg = np.asarray(clean_bg).copy()
                entity_mask = np.asarray(entity_mask)[:, :, 3] < 128
                clean_bg[entity_mask] = 0
                clean_bg = Image.fromarray(clean_bg)
                # finish the cleaning by pasting clean_bg on top of rgb
                rgb.paste(clean_bg, (0, 0), clean_bg)
                equipment_image_paths.append(equipment_image_path)

                if OCC_AUGS:
                    for _ in range(NUM_OCC_AUGS):
                        curr_rgb = deepcopy(rgb)
                        # 2) Start adding the new equipment. But mask it again using entity mask to make sure it does not exceed the entity boundaries.
                        scale_factor = random.randint(800_000, 1_400_000)
                        object_size = scale_factor / entity_depth
                        curr_equipment_image = deepcopy(equipment_image)
                        curr_equipment_image = curr_equipment_image.resize((int(object_size), int(object_size)))
                        # cut away pieces of the object
                        curr_equipment_image = transforms.RandomResizedCrop(curr_equipment_image.size, scale=(0.6, 1.0), ratio=(0.85, 1.15))(curr_equipment_image)

                        # plot equipment image into an empty canvas with the same size of the entity mask
                        equipment_image_canvas = Image.new('RGBA', (entity_mask.shape[1], entity_mask.shape[0]))
                        equipment_image_canvas.paste(curr_equipment_image, (entity_metadata['xmin'] + (entity_metadata['xmax'] - entity_metadata['xmin']) // 2 - curr_equipment_image.width // 2,
                                                                            entity_metadata['ymin'] + (entity_metadata['ymax'] - entity_metadata['ymin']) // 2 - curr_equipment_image.height // 2),
                                                     curr_equipment_image)
                        # mask the equipment image canvas using entity mask
                        equipment_image_canvas = np.asarray(equipment_image_canvas).copy()
                        equipment_image_canvas[entity_mask] = 0
                        equipment_image_canvas = Image.fromarray(equipment_image_canvas)
                        # now plot the equipment image canvas on top of the rgb image
                        curr_rgb.paste(equipment_image_canvas, (0, 0), equipment_image_canvas)

                        # crop around the instrument
                        left = entity_metadata['xmin'] + (entity_metadata['xmax'] - entity_metadata['xmin']) // 2 - curr_equipment_image.width // 2
                        top = entity_metadata['ymin'] + (entity_metadata['ymax'] - entity_metadata['ymin']) // 2 - curr_equipment_image.height // 2
                        right = left + curr_equipment_image.width
                        bottom = top + curr_equipment_image.height

                        # Define margin size (in pixels)
                        margin_size = 30  # adjust this value as needed

                        # Expand the bounding box by the margin size
                        left = max(left - margin_size, 0)
                        top = max(top - margin_size, 0)
                        right = min(right + margin_size, equipment_image_canvas.width)
                        bottom = min(bottom + margin_size, equipment_image_canvas.height)

                        # Crop the canvas around the equipment with the margin
                        bounding_box_with_margin = (left, top, right, bottom)
                        curr_rgb = curr_rgb.crop(bounding_box_with_margin)
                        view_to_rgb_augs[view_idx].append(curr_rgb)

                else:
                    curr_rgb = deepcopy(rgb)
                    # 2) Start adding the new equipment. But mask it again using entity mask to make sure it does not exceed the entity boundaries.
                    scale_factor = random.randint(800_000, 1_400_000)
                    object_size = scale_factor / entity_depth
                    equipment_image = equipment_image.resize((int(object_size), int(object_size)))

                    # plot equipment image into an empty canvas with the same size of the entity mask
                    equipment_image_canvas = Image.new('RGBA', (entity_mask.shape[1], entity_mask.shape[0]))
                    equipment_image_canvas.paste(equipment_image, (
                        entity_metadata['xmin'] + (entity_metadata['xmax'] - entity_metadata['xmin']) // 2 - equipment_image.width // 2,
                        entity_metadata['ymin'] + (entity_metadata['ymax'] - entity_metadata['ymin']) // 2 - equipment_image.height // 2),
                                                 equipment_image)
                    # mask the equipment image canvas using entity mask
                    equipment_image_canvas = np.asarray(equipment_image_canvas).copy()
                    equipment_image_canvas[entity_mask] = 0
                    equipment_image_canvas = Image.fromarray(equipment_image_canvas)
                    # now plot the equipment image canvas on top of the rgb image
                    curr_rgb.paste(equipment_image_canvas, (0, 0), equipment_image_canvas)

                    # crop around the instrument
                    left = entity_metadata['xmin'] + (entity_metadata['xmax'] - entity_metadata['xmin']) // 2 - equipment_image.width // 2
                    top = entity_metadata['ymin'] + (entity_metadata['ymax'] - entity_metadata['ymin']) // 2 - equipment_image.height // 2
                    right = left + equipment_image.width
                    bottom = top + equipment_image.height

                    # Define margin size (in pixels)
                    margin_size = 30  # adjust this value as needed

                    # Expand the bounding box by the margin size
                    left = max(left - margin_size, 0)
                    top = max(top - margin_size, 0)
                    right = min(right + margin_size, equipment_image_canvas.width)
                    bottom = min(bottom + margin_size, equipment_image_canvas.height)

                    # Crop the canvas around the equipment with the margin
                    bounding_box_with_margin = (left, top, right, bottom)
                    curr_rgb = curr_rgb.crop(bounding_box_with_margin)
                    view_to_rgb[view_idx] = curr_rgb

            # now correctly update the scene graph
            new_graph = []
            for sub, rel, obj in ref_graph:
                if sub == entity_to_replace:
                    sub = replacement_map[object_attrs['object_type']]['replace_entity_to']
                if obj == entity_to_replace:
                    obj = replacement_map[object_attrs['object_type']]['replace_entity_to']
                new_graph.append((sub, rel, obj))
            ref_graph = new_graph
            data['descriptors'][replacement_map[object_attrs['object_type']]['replace_entity_to']] = object_attrs
            data['paths'][replacement_map[object_attrs['object_type']]['replace_entity_to']] = equipment_image_paths
            data['replaced_entity'] = entity_to_replace

        # apply roles to the scene graph
        new_graph = []
        for sub, rel, obj in ref_graph:
            if sub in roles:
                sub = roles[sub][0]
            if obj in roles:
                obj = roles[obj][0]
            new_graph.append((sub, rel, obj))
        ref_graph = new_graph

        data['sg'] = ref_graph

        # 3. Now we have the scene graph, and the image. We need to save them.
        # 3.1. Save the image
        # assert at least one view is not None (1 valid crop)
        assert any([rgb is not None for rgb in view_to_rgb.values()])
        if OCC_AUGS:
            for view_idx, rgbs in view_to_rgb_augs.items():
                for aug_num, rgb in enumerate(rgbs):
                    rgb = rgb.convert('RGB')
                    rgb.save(export_path / Path(str(object_img.stem) + f"crop_cidx{view_idx}_aug{aug_num}.jpg"))
        else:
            for view_idx, rgb in view_to_rgb.items():
                if rgb is not None:  # object not visible from this view
                    rgb = rgb.convert('RGB')
                    rgb.save(export_path / Path(str(object_img.stem) + f"_crop_cidx{view_idx}.jpg"))
        return (True, d_idx)
    except Exception as e:
        return (False, d_idx)


def main():
    global list_of_syn_objects
    NUM_WORKERS = 196
    print(f'Using {NUM_WORKERS} workers')

    images_dir = Path('synthetic_or_generation/images_sdxl')
    object_type_to_images = defaultdict(list)
    for image in images_dir.glob('*.png'):
        corresponding_json = images_dir / f'{image.stem}.json'
        object_type = image.stem.split('_')[0].strip().lower()
        image_idx = int(image.stem.split('_')[-1])
        if image_idx >= 4:  # this is not mandatory
            continue
        object_type_to_images[object_type].append((image, corresponding_json))

    list_of_syn_objects = []
    for object_type, objects in object_type_to_images.items():
        for image, attrs in objects:
            list_of_syn_objects.append((object_type, image, attrs))

    # shuffle list
    random.shuffle(list_of_syn_objects)

    remaining_indices = range(len(list_of_syn_objects))

    with Pool(initializer=init_worker, processes=NUM_WORKERS) as p:
        while remaining_indices:
            print(f'Processing {len(remaining_indices)} remaining indices')
            results = list(tqdm(p.imap(main_worker, remaining_indices, chunksize=100), total=len(remaining_indices)))

            # Update remaining_indices based on the results
            remaining_indices = {d_idx for success, d_idx in results if not success}


if __name__ == '__main__':
    main()
