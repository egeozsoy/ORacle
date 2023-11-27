# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import shutil
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report
from tqdm import tqdm

from LLaVA.llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from scene_graph_prediction.scene_graph_helpers.dataset.dataset_utils import map_scene_graph_name_to_vocab_idx, map_vocab_idx_to_scene_graph_name


class OracleWrapper:
    def __init__(self, config, num_class, num_rel, weights_obj, weights_rel, relationNames,
                 model_path, model_base='liuhaotian/llava-v1.5-7b', load_8bit=False,
                 load_4bit=False):
        self.config = config
        self.mconfig = config['MODEL']
        self.n_object_types = 6
        self.weights_obj = weights_obj
        self.weights_rel = weights_rel
        self.relationNames = relationNames
        self.relation_names_lower_case = [relation.lower() for relation in self.relationNames]
        self.lr = float(self.config['LR'])
        # evaluation metrics
        self.train_take_rel_preds = defaultdict(list)
        self.train_take_rel_gts = defaultdict(list)
        self.val_take_rel_preds = defaultdict(list)
        self.val_take_rel_gts = defaultdict(list)
        self.reset_metrics()
        # Model
        disable_torch_init()

        # if model path includes directly a specific checkpoint, this won't run. So we have to modify things a little bit.
        # 1) Copy default stuff from a standard folder into the parent folder. 2) Copy stuff inside the checkpoint folder to the parent folder. 3) Change the model path to the parent folder.
        if 'checkpoint-' in model_path:
            print('Model path includes a specific checkpoint. Preparing the parent folder.')
            model_path = Path(model_path)
            defaults_path = Path('/home/guests/ege_oezsoy/Oracle/LLaVA/checkpoints/defaults')
            # copy every file from defaults_path to model_path.parent.
            for file in defaults_path.iterdir():
                if file.is_file():
                    shutil.copy(file, model_path.parent / file.name)
            # copy every file from model_path to model_path.parent
            for file in model_path.iterdir():
                if file.is_file():
                    shutil.copy(file, model_path.parent / file.name)
            # change model path to parent folder. Convert it back to string.
            model_path = model_path.parent.as_posix()

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, self.model_name, load_8bit, load_4bit)

    def forward(self, batch):

        if 'llama-2' in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()

        images = []
        for cam_idx in self.config['CAMERAS']:
            images.append(Image.open(batch['image_paths'][cam_idx - 1]).convert('RGB'))
        # Similar operation in model_worker.py
        image_tensor = process_images(images, self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # TODO this would need adapting if the prompt changes
        inp = "Describe this image using a scene graph, represented as a list of triplets. Each triplet consists of a subject(entity), an object(entity), and a predicate. Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, operating, preparing, sawing, suturing, touching]."
        # inp = 'Describe the main action in this scene.' # TODO remove this. This only uses the main action
        # first message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                use_cache=True,
                max_new_tokens=300,
                stopping_criteria=[stopping_criteria]
            )

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        return outputs

    def reset_metrics(self, split=None):
        if split == 'train':
            self.train_take_rel_preds = defaultdict(list)
            self.train_take_rel_gts = defaultdict(list)
        elif split == 'val':
            self.val_take_rel_preds = defaultdict(list)
            self.val_take_rel_gts = defaultdict(list)
        else:
            self.train_take_rel_preds = defaultdict(list)
            self.train_take_rel_gts = defaultdict(list)
            self.val_take_rel_preds = defaultdict(list)
            self.val_take_rel_gts = defaultdict(list)

    def update_metrics(self, batch, rel_pred, split='train'):
        if split == 'train':
            self.train_take_rel_preds[batch['take_idx']].extend(rel_pred.detach().cpu().numpy().argmax(1))
            self.train_take_rel_gts[batch['take_idx']].extend(batch['gt_rels'].detach().cpu().numpy())
        elif split == 'val':
            self.val_take_rel_preds[batch['take_idx']].extend(rel_pred.detach().cpu().numpy().argmax(1))
            self.val_take_rel_gts[batch['take_idx']].extend(batch['gt_rels'].detach().cpu().numpy())
        else:
            raise NotImplementedError()

    def predict(self, batch, batch_idx, dataloader_idx=0):
        # TODO redo this
        raise NotImplementedError()
        obj_pred, rel_pred, _, _, _, _, probs = self(batch, return_meta_data=True)
        predicted_relations = torch.max(rel_pred.detach(), 1)[1]
        all_scores = F.softmax(rel_pred, dim=1)

        # Get the scores that correspond to predicted_relations
        # scores = all_scores[range(rel_pred.shape[0]), predicted_relations]
        relations = []
        for idy, (edge, rel) in enumerate(zip(batch['edge_indices'].transpose(0, 1), predicted_relations)):
            if rel == self.relationNames.index('none'):
                continue
            start = edge[0]
            end = edge[1]
            start_name = batch['objs_json'][start.item() + 1]
            end_name = batch['objs_json'][end.item() + 1]
            rel_name = self.relationNames[rel]
            # print(f'{start_name} -> {rel_name} -> {end_name}')
            # if output_scores: relations.append((start_name, rel_name, end_name, scores[idy].item()))
            relations.append((start_name, rel_name, end_name))

        return (batch['scan_id'], relations)

    # def test_step(self, batch, batch_idx): # not for inference
    #     return self.validation_step(batch, batch_idx)

    @torch.no_grad()
    def compute_optimal_human_indices(self, pred, gt):
        '''
        Matches the human indices predicted by the model(NOT THE PATIENT) to the ground truth to lead the highest accuracy
        Human indices: [7,8,9,10,11]
        To simplify:
        - If GT does not include a human index, don't assign this in the predictions
        - (If PT does not include a human index, don't iterate over it)
        '''
        if (len(gt) == 0 or len(pred) == 0):
            return {}
        gt_human_indices = set(gt[torch.logical_and(gt >= 7, gt <= 11)].tolist())
        pred_argmax = torch.tensor(pred).flatten()  # In eval, it is already argmaxed
        pred_human_indices = set(pred_argmax[torch.logical_and(pred_argmax >= 7,
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

    def validate(self, dataloader, limit_val_batches=None):
        take_rel_preds = defaultdict(list)
        take_rel_gts = defaultdict(list)
        # if limit_val_batches is int, then limit the number of batches to this number, if float, then limit the number of batches to this fraction of the total number of batches.
        limit_counter = None
        if isinstance(limit_val_batches, int):
            limit_counter = limit_val_batches
        elif isinstance(limit_val_batches, float):
            limit_counter = int(limit_val_batches * len(dataloader))

        for batch in tqdm(dataloader):
            if limit_counter is not None:
                if limit_counter <= 0:
                    break
                limit_counter -= 1
            output = self.forward(batch)
            triplet_str = output.split(';')
            triplets = []
            human_roles = set()  # Need to be mapped for this evaluation
            for triplet in triplet_str:
                triplet = triplet.replace('.', '').strip()
                if triplet == '':
                    continue
                triplet = triplet.split(',')
                triplet = [elem.strip() for elem in triplet]
                if len(triplet) != 3:
                    continue
                sub, obj, pred = triplet
                if sub == 'anesthetist':
                    sub = 'anaesthetist'
                if obj == 'anesthetist':
                    obj = 'anaesthetist'
                if sub in ['head surgeon', 'assistant surgeon', 'circulator', 'nurse', 'anaesthetist']:
                    human_roles.add(sub)
                if obj in ['head surgeon', 'assistant surgeon', 'circulator', 'nurse', 'anaesthetist']:
                    human_roles.add(obj)
                triplets.append((sub, pred, obj))
            # these have to be mapped. First to human names, also the predicates
            human_roles_to_indices = {human_role: f'human_{idx}' for idx, human_role in enumerate(sorted(human_roles))}
            rel_preds = []
            for (sub, pred, obj) in triplets:
                try:
                    sub = map_scene_graph_name_to_vocab_idx(human_roles_to_indices.get(sub, sub).replace(' ', '_'))
                    obj = map_scene_graph_name_to_vocab_idx(human_roles_to_indices.get(obj, obj).replace(' ', '_'))
                    pred = map_scene_graph_name_to_vocab_idx(pred)
                    rel_preds.append((sub, pred, obj))
                except Exception as e:
                    print(e)
                    continue
            rel_labels = torch.tensor(batch['relations_tokenized'])
            human_idx_map = self.compute_optimal_human_indices(rel_preds, rel_labels)
            rel_preds = [(human_idx_map.get(pred_sub, pred_sub), pred_rel, human_idx_map.get(pred_obj, pred_obj)) for pred_sub, pred_rel, pred_obj in
                         rel_preds]  # map
            human_readable_pred = [(map_vocab_idx_to_scene_graph_name(sub), map_vocab_idx_to_scene_graph_name(pred), map_vocab_idx_to_scene_graph_name(obj))
                                   for sub, pred, obj in rel_preds]
            human_readable_gt = [(map_vocab_idx_to_scene_graph_name(sub), map_vocab_idx_to_scene_graph_name(pred), map_vocab_idx_to_scene_graph_name(obj))
                                 for sub, pred, obj in rel_labels.tolist()]

            if len(rel_labels) == 0:
                all_gt_objects = []
            else:
                all_gt_objects = sorted(set(rel_labels[:, [0, 2]].flatten().tolist()))
            # Search for all possible relationships between objects, those that don't have any should be labeled 'none', otherwise the correct relation is asked for

            for gt_obj1 in all_gt_objects:
                for gt_obj2 in all_gt_objects:
                    if gt_obj1 == gt_obj2:
                        continue
                    for gt_sub, gt_rel, gt_obj in rel_labels:
                        if gt_sub == gt_obj1 and gt_obj == gt_obj2:
                            take_rel_gts[batch['take_idx']].append(self.relation_names_lower_case.index(map_vocab_idx_to_scene_graph_name(gt_rel.item())))
                            break
                    else:
                        take_rel_gts[batch['take_idx']].append(self.relation_names_lower_case.index('none'))
                    for pred_sub, pred_rel, pred_obj in rel_preds:
                        if pred_sub == gt_obj1 and pred_obj == gt_obj2:
                            try:
                                pred_rel_id = self.relation_names_lower_case.index(map_vocab_idx_to_scene_graph_name(pred_rel))
                            except Exception as e:  # if a   none sense relation was predicted ignore
                                pred_rel_id = self.relation_names_lower_case.index('none')
                            take_rel_preds[batch['take_idx']].append(pred_rel_id)
                            break
                    else:
                        take_rel_preds[batch['take_idx']].append(self.relation_names_lower_case.index('none'))

        self.val_take_rel_preds, self.val_take_rel_gts = take_rel_preds, take_rel_gts
        self.evaluate_predictions(None, 'val')
        self.reset_metrics(split='val')

    # def test_epoch_end(self, outputs):
    #     return self.validation_epoch_end(outputs)

    def evaluate_predictions(self, epoch_loss, split):
        if split == 'train':
            take_rel_preds = self.train_take_rel_preds
            take_rel_gts = self.train_take_rel_gts
        elif split == 'val':
            take_rel_preds = self.val_take_rel_preds
            take_rel_gts = self.val_take_rel_gts
        else:
            raise NotImplementedError()

        all_rel_gts = []
        all_rel_preds = []
        for take_idx in sorted(take_rel_preds.keys()):
            rel_preds = take_rel_preds[take_idx]
            rel_gts = take_rel_gts[take_idx]
            all_rel_gts.extend(rel_gts)
            all_rel_preds.extend(rel_preds)

            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames)
            print(f'\nTake {take_idx}\n')
            print(cls_report)

        results = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                        target_names=self.relationNames, output_dict=True)
        macro_f1 = results['macro avg']['f1-score']
        print(f'{split} Results:\n')
        cls_report = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                           target_names=self.relationNames)
        print(cls_report)
        return macro_f1
