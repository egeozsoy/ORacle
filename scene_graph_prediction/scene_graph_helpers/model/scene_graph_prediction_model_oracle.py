# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import json
import re
from collections import defaultdict
from copy import deepcopy

import torch
from PIL import Image
from sklearn.metrics import classification_report
from tqdm import tqdm

from LLaVA.llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, VIS_DESCRIPTOR_TOKEN
from LLaVA.llava.conversation import SeparatorStyle, default_conversation
from LLaVA.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from LLaVA.llava.model.builder import load_pretrained_model
from scene_graph_prediction.llava_helpers.scene_graph_converters import llava_sg_to_surgery_sg, surgery_sg_to_memory_str
from scene_graph_prediction.scene_graph_helpers.dataset.dataset_utils import map_scene_graph_name_to_vocab_idx, map_vocab_idx_to_scene_graph_name, reversed_role_synonyms


class OracleWrapper:
    def __init__(self, config, num_class, num_rel, weights_obj, weights_rel, relationNames,
                 model_path, model_base='liuhaotian/llava-v1.5-7b', load_8bit=False,
                 load_4bit=False, mv_desc=False):
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

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, self.model_name, load_8bit, load_4bit)
        self.model.config.mv_type = self.mconfig['mv_type']
        self.model.config.tokenizer_padding_side = "left"
        self.temporal_online_prediction = False
        self.mv_desc = mv_desc
        if 'temporality' in config and config['temporality'] == 'GT':
            print('Loading temporality GT')
            self.take_timepoint_to_memory_str = {}
            with open('data/llava_samples/train_50perm_Truetemp_doublemem_Truetempaug_longshort_doublesg_take_timepoint_to_memory_str.json') as f:  # TODO adapt this
                self.take_timepoint_to_memory_str = json.load(f)
            with open('data/llava_samples/val_50perm_Truetemp_doublemem_Truetempaug_longshort_doublesg_take_timepoint_to_memory_str.json') as f:
                self.take_timepoint_to_memory_str.update(json.load(f))
        elif 'temporality' in config and config['temporality'] == 'PRED':
            print('Preparing temporality PRED')
            self.take_to_history = defaultdict(list)
            self.temporal_online_prediction = True

    def forward(self, batch):
        batchsize = len(batch)
        all_images = []
        all_prompts = []
        all_vis_descriptor_embs = []
        for elem in batch:
            conv = deepcopy(default_conversation)

            images = []
            for cam_idx in self.config['CAMERAS']:
                images.append(Image.open(elem['image_paths'][cam_idx - 1]).convert('RGB'))
            # Similar operation in model_worker.py
            image_tensor = process_images(images, self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.bfloat16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.bfloat16)
            all_images.append(image_tensor)

            # TODO this would need adapting if the prompt changes
            # inp = "Describe this image using a scene graph, represented as a list of triplets. Each triplet consists of a subject(entity), an object(entity), and a predicate. Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching]."
            # inp = "Describe this image at timepoint T using a scene graph, represented as a list of triplets. Each triplet consists of a subject(entity), an object(entity), and a predicate. Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching]."
            # inp = 'Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text. Do not include the timepoint format "T-" in the triplets.'
            if '_symbolic' in self.model_name:
                # inp = 'Entities: [A, B, C, D, E, F, G, H, I, J, K]. Predicates: [α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ]. <knowledge_start> A: Primary operator in surgery, handles critical tasks. B: Supports head surgeon, assists in surgical tasks. C: Coordinates OR activities and tools. D: Assists in surgical prep and recovery. E: Administers anesthesia, monitors patient. F: Undergoes surgical procedure. G: Metallic object sometimes covered with sheets. Holds surgical instruments. H: Central table in OR for patient. I: Holds additional surgical supplies, auxiliary to instrument table. J: Contains devices for anesthesia administration. K: Tools for performing surgical tasks. α: Collaboration between staff. β: Process of affixing knee implants. γ: Sanitization of OR environment and equipment. δ: Proximity of medical staff or equipment to each other in OR. ε: use of scalpel for incisions on patient. ζ: Utilizing an orange drill in surgery. η: use of a hammer with wooden holder and gray head in surgery. θ: Grasping surgical instruments. ι: Patient positioned on the operating table. κ: Handling of medical objects like operating tables or anesthesia machines. λ: Includes draping and sterilization. μ: use of a green/gray saw for surgical procedures on patient. ν: Stitching using medical scissors. ξ: Physical contact between entities. <knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
                inp = 'Entities: [A, B, C, D, E, F, G, H, I, J, K]. Predicates: [α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ]. <knowledge_start> A: primary operator in surgery, handles critical tasks. B: supports head surgeon, assists in surgical tasks. C: coordinates OR activities and tools. D: assists in surgical prep and recovery. E: administers anesthesia, monitors patient. F: undergoes surgical procedure. G: instrument table, blue, large, rectangular, matte. H: operating table, black, large, rectangular, rubber. I: secondary table, gray, large, rectangular, metal. J: anesthesia equipment, white, large, irregular, matte. K: instrument, handheld. α: collaboration between staff. β: use of a tool: black, handheld, straight, metal, surgical bone cement gun. γ: use of a tool: white, handheld, irregular, sanitization equipment. δ: proximity of medical staff or equipment to each other in OR. ε: use of a tool: white, small, straight, plastic, surgical scalpel. ζ: use of a tool: orange, handheld, L-shape, plastic, surgical drill. η: use of a tool: brown, handheld, T-shape, metal, surgical hammer. θ: grasping surgical instruments. ι: patient positioned on the operating table. κ: handling of medical objects like operating tables or anesthesia machines. λ: includes draping and sterilization. μ: use of a tool: green, handheld, round, plastic, surgical bone saw. ν: use of a tool: gray, small, straight, metal, surgical scissors. ξ: physical contact between entities. <knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
            else:
                inp = 'Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'

            if 'USE_VIS_DESC' in self.config and self.config['USE_VIS_DESC'] == True:
                # order of img_patches: anesthesia machine, cementing, cutting, drilling, hammering, sawing, suturing
                if not self.mv_desc:
                    inp = f'Entities: [A, B, C, D, E, F, G, H, I, J, K]. Predicates: [α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ]. <knowledge_start> A: primary operator in surgery, handles critical tasks. B: supports head surgeon, assists in surgical tasks. C: coordinates OR activities and tools. D: assists in surgical prep and recovery. E: administers anesthesia, monitors patient. F: undergoes surgical procedure. G: instrument table, blue, large, rectangular, matte. H: operating table, black, large, rectangular, rubber. I: secondary table, gray, large, rectangular, metal. J: {VIS_DESCRIPTOR_TOKEN}. K: Instrument, handheld. α: collaboration between staff. β: use of a tool: {VIS_DESCRIPTOR_TOKEN}. γ: use of a tool: white, handheld, irregular, sanitization equipment. δ: proximity of medical staff or equipment to each other in OR. ε: use of a tool: {VIS_DESCRIPTOR_TOKEN}. ζ: use of a tool: {VIS_DESCRIPTOR_TOKEN}. η: use of a tool: {VIS_DESCRIPTOR_TOKEN}. θ: grasping surgical instruments. ι: patient positioned on the operating table. κ: handling of medical objects like operating tables or anesthesia machines. λ: includes draping and sterilization. μ: use of a tool: {VIS_DESCRIPTOR_TOKEN}. ν: use of a tool: {VIS_DESCRIPTOR_TOKEN}. ξ: physical contact between entities. <knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
                else:
                    inp = f'Entities: [A, B, C, D, E, F, G, H, I, J, K]. Predicates: [α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ]. <knowledge_start> A: primary operator in surgery, handles critical tasks. B: supports head surgeon, assists in surgical tasks. C: coordinates OR activities and tools. D: assists in surgical prep and recovery. E: administers anesthesia, monitors patient. F: undergoes surgical procedure. G: instrument table, blue, large, rectangular, matte. H: operating table, black, large, rectangular, rubber. I: secondary table, gray, large, rectangular, metal. J: {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN}. K: Instrument, handheld. α: collaboration between staff. β: use of a tool: {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN}. γ: use of a tool: white, handheld, irregular, sanitization equipment. δ: proximity of medical staff or equipment to each other in OR. ε: use of a tool: {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN}. ζ: use of a tool: {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN}. η: use of a tool: {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN}. θ: grasping surgical instruments. ι: patient positioned on the operating table. κ: handling of medical objects like operating tables or anesthesia machines. λ: includes draping and sterilization. μ: use of a tool: {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN}. ν: use of a tool: {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN} {VIS_DESCRIPTOR_TOKEN}. ξ: physical contact between entities. <knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
                vis_desc = [des.to(self.model.device, dtype=torch.bfloat16) for des in elem['vis_descriptor_embs']]
                all_vis_descriptor_embs.append(vis_desc)

            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            if 'temporality' in self.config:
                if self.config['temporality'] == 'GT':
                    take_idx, timepoint_idx, _ = elem['scan_id'].split('_')
                    take_idx = int(take_idx)
                    timepoint_idx = int(timepoint_idx)
                    take_timepoint = f'{take_idx}_{timepoint_idx}'
                    memory_str = self.take_timepoint_to_memory_str[take_timepoint]
                elif self.config['temporality'] == 'PRED':
                    take_idx = elem['take_idx']
                    timepoint_idx = int(elem['scan_id'].split('_')[1])
                    raw_triplets = self.take_to_history[take_idx]
                    surgery_sg_triplets = llava_sg_to_surgery_sg(raw_triplets, entity_of_interest=None, IRRELEVANT_PREDS=['closeto', 'closeTo'])
                    surgery_sg_triplets = [elem for elem in surgery_sg_triplets if elem[0] < timepoint_idx]
                    memory_str = surgery_sg_to_memory_str(surgery_sg_triplets, current_timepoint=timepoint_idx, TEMPORAL_STYLE='longshort')
                else:
                    raise NotImplementedError()
                if len(memory_str) > 5000:
                    print(f'Warning: memory string is too long ({len(memory_str)} chars)')
                    memory_str = '...' + memory_str[-5000:]
                # inp = inp.replace(f'{DEFAULT_IMAGE_TOKEN}\n', f'{DEFAULT_IMAGE_TOKEN}\nMemory: {memory_str}.')
                inp = inp.replace(f'{DEFAULT_IMAGE_TOKEN}\n', f'{DEFAULT_IMAGE_TOKEN}\n<memory_start>: {memory_str}<memory_end>.\n')
            conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            all_prompts.append(prompt)

        if batchsize == 1:
            input_ids = tokenizer_image_token(all_prompts[0], self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
            image_tensor = all_images[0]

        else:
            input_ids = [tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in all_prompts]
            # merge with left padding
            inverted_input_ids = [torch.flip(input_id, dims=[0]) for input_id in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(inverted_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            # invert back
            input_ids = torch.flip(input_ids, dims=[1]).to(self.model.device)
            image_tensor = torch.cat(all_images)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            # print(f'Length of input_ids: {input_ids.shape[1]}')
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                use_cache=True,
                max_new_tokens=300,
                stopping_criteria=[stopping_criteria],
                vis_descriptor_embs=all_vis_descriptor_embs if 'USE_VIS_DESC' in self.config and self.config['USE_VIS_DESC'] else None
            )
        if batchsize == 1:
            outputs = [self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:].tolist(), skip_special_tokens=True)

        if '_symbolic' in self.model_name or self.config['USE_VIS_DESC']:
            # parse outputs back to head surgeon etc..
            symbolic_parsed = []
            # TODO this might need updating
            replace_map = {'A': 'head surgeon', 'B': 'assistant surgeon', 'C': 'circulator', 'D': 'nurse', 'E': 'anaesthetist', 'F': 'patient', 'G': 'instrument table', 'H': 'operating table',
                           'I': 'secondary table', 'J': 'anesthesia equipment', 'K': 'instrument', 'α': 'assisting', 'β': 'cementing', 'γ': 'cleaning', 'δ': 'closeTo', 'ε': 'cutting', 'ζ': 'drilling',
                           'η': 'hammering', 'θ': 'holding', 'ι': 'lyingOn', 'κ': 'manipulating', 'λ': 'preparing', 'μ': 'sawing', 'ν': 'suturing', 'ξ': 'touching'}
            # we use regex to replace all occurences of the symbols
            regex = re.compile("|".join(map(re.escape, replace_map.keys())))
            for output in outputs:
                # map everything according to replace_map. this replacement has to be done all at once, otherwise we might replace a symbol that was already replaced.
                cleaned_output = output.replace('<SG>', '').replace('</SG>', '')
                cleaned_output = regex.sub(lambda match: replace_map[match.group(0)], cleaned_output)
                cleaned_output = f'<SG>{cleaned_output}</SG>'
                symbolic_parsed.append(cleaned_output)

            outputs = symbolic_parsed
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

    def infer(self, dataloader):
        return self.validate(dataloader, return_raw_predictions=True)

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

    def validate(self, dataloader, limit_val_batches=None, logging_information=None, return_raw_predictions=False):
        take_rel_preds = defaultdict(list)
        take_rel_gts = defaultdict(list)
        scan_id_to_raw_predictions = {}  # dictionary to store predicted scene graphs
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

            # if self.temporal_online_prediction than assert that batchsize is 1 and sorted. But we can only assert for batchsize 1
            assert len(batch) == 1 or not self.temporal_online_prediction

            outputs = self.forward(batch)
            for idx, output in enumerate(outputs):
                elem = batch[idx]
                timepoint = int(elem['scan_id'].split('_')[1])
                take_idx = elem['take_idx']
                # remove everything between the first """ and the last """ using regex. This is used for chain of thought
                output = re.sub(r'""".*?"""', '', output, flags=re.DOTALL)
                if '<SG>' in output and '</SG>' in output and output.index('<SG>') < output.index('</SG>'):
                    triplet_str = output.split('<SG>')[1].split('</SG>')[0].strip().split(';')
                else:
                    triplet_str = output.split(';')
                triplets = []
                raw_triplets = []
                human_roles = set()  # Need to be mapped for this evaluation
                for triplet in triplet_str:
                    triplet = triplet.replace('.', '').replace('</s>', '').replace('<s>', '').strip()
                    if triplet == '':
                        continue
                    triplet = triplet.split(',')
                    triplet = [elem.strip() for elem in triplet]
                    if len(triplet) != 3:
                        continue
                    sub, obj, pred = triplet
                    raw_triplets.append((sub, pred, obj))
                    if sub in reversed_role_synonyms:
                        sub = reversed_role_synonyms[sub]
                    if obj in reversed_role_synonyms:
                        obj = reversed_role_synonyms[obj]
                    if sub in ['head surgeon', 'assistant surgeon', 'circulator', 'nurse', 'anaesthetist']:
                        human_roles.add(sub)
                    if obj in ['head surgeon', 'assistant surgeon', 'circulator', 'nurse', 'anaesthetist']:
                        human_roles.add(obj)
                    triplets.append((sub, pred, obj))
                # these have to be mapped. First to human names, also the predicates
                scan_id_to_raw_predictions[elem['scan_id']] = raw_triplets
                if self.temporal_online_prediction:
                    # TODO we could shuffle the raw triplets?
                    self.take_to_history[take_idx].append({'timepoint_idx': timepoint, 'scene_graph': raw_triplets})
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
                rel_labels = torch.tensor(batch[idx]['relations_tokenized'])
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
                                take_rel_gts[batch[idx]['take_idx']].append(self.relation_names_lower_case.index(map_vocab_idx_to_scene_graph_name(gt_rel.item())))
                                break
                        else:
                            take_rel_gts[batch[idx]['take_idx']].append(self.relation_names_lower_case.index('none'))
                        for pred_sub, pred_rel, pred_obj in rel_preds:
                            if pred_sub == gt_obj1 and pred_obj == gt_obj2:
                                try:
                                    pred_rel_id = self.relation_names_lower_case.index(map_vocab_idx_to_scene_graph_name(pred_rel))
                                except Exception as e:  # if a   none sense relation was predicted ignore
                                    pred_rel_id = self.relation_names_lower_case.index('none')
                                take_rel_preds[batch[idx]['take_idx']].append(pred_rel_id)
                                break
                        else:
                            take_rel_preds[batch[idx]['take_idx']].append(self.relation_names_lower_case.index('none'))

        self.val_take_rel_preds, self.val_take_rel_gts = take_rel_preds, take_rel_gts
        self.evaluate_predictions(None, 'val', logging_information=logging_information)
        self.reset_metrics(split='val')

        if return_raw_predictions:
            return scan_id_to_raw_predictions

    # def test_epoch_end(self, outputs):
    #     return self.validation_epoch_end(outputs)

    def evaluate_predictions(self, epoch_loss, split, logging_information=None):
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
                                               target_names=self.relationNames, output_dict=True)
            for rel_name in self.relationNames:
                for score_type in ['precision', 'recall', 'f1-score']:
                    # self.log(f'{rel_name}/{take_idx}_{score_type[:2].upper()}', cls_report[rel_name][score_type], rank_zero_only=True)
                    if logging_information is not None:
                        logging_information['logger'].log_metrics({f'{rel_name}/{take_idx}_{score_type[:2].upper()}': cls_report[rel_name][score_type]}, step=logging_information['checkpoint_id'])

            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames)
            print(f'\nTake {take_idx}\n')
            print(cls_report)

        results = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                        target_names=self.relationNames, output_dict=True)
        macro_f1 = results['macro avg']['f1-score']
        if logging_information is not None:
            # logging_information will have a key use it to log to wandb. It will also have a checkpoint int, which we also want to log (similar to epoch). Also we want to use the split to log as train or val
            logging_information['logger'].log_metrics({f'{logging_information["split"]}_precision': results['macro avg']['precision']}, step=logging_information['checkpoint_id'])
            logging_information['logger'].log_metrics({f'{logging_information["split"]}_recall': results['macro avg']['recall']}, step=logging_information['checkpoint_id'])
            logging_information['logger'].log_metrics({f'{logging_information["split"]}_macro_f1': results['macro avg']['f1-score']}, step=logging_information['checkpoint_id'])

        print(f'{split} Results:\n')
        cls_report = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                           target_names=self.relationNames)
        print(cls_report)
        return macro_f1
