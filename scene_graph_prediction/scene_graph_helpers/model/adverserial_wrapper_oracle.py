import random
import re
from copy import deepcopy

import torch
from PIL import Image
from sklearn.metrics import classification_report
from tqdm import tqdm

from LLaVA.llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, VIS_DESCRIPTOR_TOKEN
from LLaVA.llava.conversation import SeparatorStyle, default_conversation
from LLaVA.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from LLaVA.llava.model.builder import load_pretrained_model
from scene_graph_prediction.llava_helpers.descriptors import ENTITY_SYMBOLS, PREDICATE_SYMBOLS, ENTITY_DESCRIPTORS, PREDICATE_DESCRIPTORS


class AdverserialOracleWrapper:
    def __init__(self, config, labels, model_path, model_base='liuhaotian/llava-v1.5-7b', load_8bit=False, load_4bit=False):
        self.config = config
        self.mconfig = config['MODEL']
        self.labels = labels

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, self.model_name, load_8bit, load_4bit)
        self.model.config.mv_type = self.mconfig['mv_type']
        self.model.config.tokenizer_padding_side = "left"

        if self.config['USE_VIS_DESC']:
            self.vis_knowledge_paths = {
                'anesthesia equipment':'data/original_crops/anesthesia equipment_take1.pt',
                'cementing':'data/original_crops/cementing_take1.pt',
                'cutting':'data/original_crops/cutting_take1.pt',
                'drilling':'data/original_crops/drilling_take1.pt',  # 'data/original_crops/test_crops/green_drill_crop.pt',
                'hammering':'data/original_crops/hammering_take1.pt',
                'sawing':'data/original_crops/sawing_take1.pt',  # 'data/original_crops/test_crops/orange_saw_crop.pt'
                'suturing':'data/original_crops/suturing_take1.pt'
            }

            self.vis_descriptor_embs = {}
            for name, vis_knowledge_path in self.vis_knowledge_paths.items():
                vis_descriptor_emb = torch.load(vis_knowledge_path, map_location='cpu')
                self.vis_descriptor_embs[vis_knowledge_path] = vis_descriptor_emb

    def _symbolic_prompt_maker(self, customizations):
        '''
        Custimizations is simply a list of tuples, where the first element is either an entity or a predicate, and the second element is the description of that entity or predicate, and the third indicates if it is an entity or a predicate. If it is an entity, the third element is 'e', if it is a predicate, the third element is 'p'.
        '''
        entity_name_to_symbol = {}
        entity_symbol_to_descriptor = {}
        predicate_name_to_symbol = {}
        predicate_symbol_to_descriptor = {}
        entity_symbols = ENTITY_SYMBOLS.copy()
        predicate_symbols = PREDICATE_SYMBOLS.copy()
        entity_descriptors = ENTITY_DESCRIPTORS.copy()
        predicate_descriptors = PREDICATE_DESCRIPTORS.copy()

        # define which entities and predicates from original objects are visual prompts
        visual_entities = ["anesthesia equipment"]
        visual_predicates = ["cementing","cutting","drilling","hammering","sawing","suturing"]

        # change descriptors of visual prompts
        for entity in visual_entities:
            entity_descriptors[entity] = [f'{VIS_DESCRIPTOR_TOKEN}.']
        for predicate in visual_predicates:
            predicate_descriptors[predicate] = [f'use of a tool: {VIS_DESCRIPTOR_TOKEN}.']

        # adding any new entities with corresponding descriptions, or overwriting existing ones. Same for predicates. The rest is automatically handled.
        vis_knowledge_paths_copy = deepcopy(self.vis_knowledge_paths)
        for name, descriptor, type, vis_desc_path in customizations:
            if type == 'e':
                entity_descriptors[name] = [descriptor]
                if name not in visual_entities:
                    visual_entities.append(name)
                vis_knowledge_paths_copy[name] = str(vis_desc_path).replace(".jpg", "_crop.pt")
            elif type == 'p':
                predicate_descriptors[name] = [descriptor]
                if name not in visual_predicates:
                    visual_predicates.append(name)
                vis_knowledge_paths_copy[name] = str(vis_desc_path).replace(".jpg", "_crop.pt")
            else:
                raise NotImplementedError()

        sym_to_descriptor_paths = {}
        for entity_name, descriptors in entity_descriptors.items():
            entity_name_to_symbol[entity_name] = entity_symbols.pop(0)
            entity_symbol_to_descriptor[entity_name_to_symbol[entity_name]] = descriptors[0]
            if entity_name in visual_entities:
                sym_to_descriptor_paths[entity_name_to_symbol[entity_name]] = vis_knowledge_paths_copy[entity_name]
        for predicate_name, descriptors in predicate_descriptors.items():
            predicate_name_to_symbol[predicate_name] = predicate_symbols.pop(0)
            predicate_symbol_to_descriptor[predicate_name_to_symbol[predicate_name]] = descriptors[0]
            if predicate_name in visual_predicates:
                sym_to_descriptor_paths[predicate_name_to_symbol[predicate_name]] = vis_knowledge_paths_copy[predicate_name]

        entity_symbol_to_descriptor_sorted = sorted(entity_symbol_to_descriptor.items(), key=lambda x: x[0])
        predicate_symbol_to_descriptor_sorted = sorted(predicate_symbol_to_descriptor.items(), key=lambda x: x[0])
        entity_symbols = ", ".join([elem[0] for elem in entity_symbol_to_descriptor_sorted])
        predicate_symbols = ", ".join([elem[0] for elem in predicate_symbol_to_descriptor_sorted])
        human_prompt = f'Entities: [{entity_symbols}]. Predicates: [{predicate_symbols}]. <knowledge_start> '

        vis_knowledge_paths = [] # fill in correct order
        for entity_symbol, descriptor in entity_symbol_to_descriptor_sorted:
            human_prompt += f'{entity_symbol}: {descriptor} '
            if entity_symbol in sym_to_descriptor_paths:
                vis_knowledge_paths.append(sym_to_descriptor_paths[entity_symbol])
        for predicate_symbol, descriptor in predicate_symbol_to_descriptor_sorted:
            human_prompt += f'{predicate_symbol}: {descriptor} '
            if predicate_symbol in sym_to_descriptor_paths:
                vis_knowledge_paths.append(sym_to_descriptor_paths[predicate_symbol])
        human_prompt += f'<knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'

        return human_prompt, entity_name_to_symbol, predicate_name_to_symbol, vis_knowledge_paths

    def forward(self, batch, ):
        assert len(batch) == 1
        elem = batch[0]
        image_path = elem['image_path']
        test_condition = elem['test_condition']
        vis_desc_path = elem['vis_desc_path']
        textual_attributes = elem['textual_attributes']
        label = elem['label']

        conv = deepcopy(default_conversation)

        images = [Image.open(image_path).convert('RGB')]
        # Similar operation in model_worker.py
        image_tensor = process_images(images, self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.bfloat16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.bfloat16)

        if 'USE_VIS_DESC' in self.config and self.config['USE_VIS_DESC'] == True:
            if label in ['cementing', 'cutting', 'drilling', 'hammering', 'robotic_sawing', 'sawing', 'suturing']: #predicate
                if label == 'robotic_sawing':
                    label = 'sawing'
                customizations = [(label, f'use of a tool: {VIS_DESCRIPTOR_TOKEN}.', 'p', vis_desc_path)]
            elif label in ['kuka', 'mako']:
                customizations = [(label, f'{VIS_DESCRIPTOR_TOKEN}.', 'e', vis_desc_path)]
            else:
                raise NotImplementedError(label)

            inp, entity_name_to_symbol, predicate_name_to_symbol, vis_knowledge_paths = self._symbolic_prompt_maker(customizations)
        elif '_symbolic' in self.model_name:
            textual_str = f'{textual_attributes["color"]}, {textual_attributes["size"]}, {textual_attributes["shape"]}, {textual_attributes["texture"]}, {textual_attributes["object_type"]}.'
            if label in ['cementing', 'cutting', 'drilling', 'hammering', 'robotic_sawing', 'sawing', 'suturing']:
                if label == 'robotic_sawing':
                    label = 'sawing'
                customizations = [(label, f'use of a tool: {textual_str}', 'p', None)]
            elif label in ['kuka', 'mako']:
                customizations = [(label, f'{textual_str}', 'e', None)]
            else:
                raise NotImplementedError(label)
            inp, entity_name_to_symbol, predicate_name_to_symbol, vis_knowledge_paths = self._symbolic_prompt_maker(customizations)
            vis_knowledge_paths = None
        else:
            inp = 'Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
            vis_knowledge_paths = None

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

        if vis_knowledge_paths is not None:
            vis_descriptor_embs = []
            for vis_knowledge_path in vis_knowledge_paths:
                if vis_knowledge_path in self.vis_descriptor_embs:
                    vis_descriptor_emb = self.vis_descriptor_embs[vis_knowledge_path]
                else:
                    vis_descriptor_emb = torch.load(vis_knowledge_path, map_location='cpu')
                vis_descriptor_embs.append(vis_descriptor_emb)
            vis_descriptor_embs = [des.to(self.model.device, dtype=torch.bfloat16) for des in vis_descriptor_embs]
        else:
            vis_descriptor_embs = None

        with torch.inference_mode():
            # print(f'Length of input_ids: {input_ids.shape[1]}')
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                use_cache=True,
                max_new_tokens=300,
                stopping_criteria=[stopping_criteria],
                vis_descriptor_embs=vis_descriptor_embs if 'USE_VIS_DESC' in self.config and self.config['USE_VIS_DESC'] else None
            )
        outputs = [self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()]

        if '_symbolic' in self.model_name or self.config['USE_VIS_DESC']:
            symbolic_parsed = []

            replace_map = {symbol: name for name, symbol in entity_name_to_symbol.items() | predicate_name_to_symbol.items()}
            # we use regex to replace all occurences of the symbols
            regex = re.compile("|".join(map(re.escape, replace_map.keys())))
            for output in outputs:
                # map everything according to replace_map. this replacement has to be done all at once, otherwise we might replace a symbol that was already replaced.
                cleaned_output = output.replace('<SG>', '').replace('</SG>', '')
                cleaned_output = regex.sub(lambda match: replace_map[match.group(0)], cleaned_output)
                cleaned_output = f'<SG>{cleaned_output}</SG>'
                symbolic_parsed.append(cleaned_output)

            outputs = symbolic_parsed

        output = outputs[0]

        # now we test the test case
        test_condition_fullfilled = test_condition.lower().replace('<', '').replace('>', '') in output.lower()
        return test_condition_fullfilled, output

    def validate(self, dataloader):
        labels_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        # extend it with "none" label
        labels_to_idx['none'] = len(labels_to_idx)
        preds = []
        gts = []

        for batch in tqdm(dataloader):
            assert len(batch) == 1
            elem = batch[0]
            is_pos = elem['is_pos']
            label_idx = labels_to_idx[elem['label']]

            test_condition_fullfilled, output = self.forward(batch)  # if is_pos it should be fullfilled, if not it should not be fullfilled
            print(elem['label'], is_pos, test_condition_fullfilled)
            if is_pos == True:
                gts.append(label_idx)
            else:
                gts.append(labels_to_idx['none'])
            if test_condition_fullfilled:
                preds.append(label_idx)
            else:
                preds.append(labels_to_idx['none'])

        included_labels = list(range(len(labels_to_idx) - 1))  # Excluding the last label ('none')

        results = classification_report(gts, preds, labels=included_labels,
                                        target_names=self.labels, output_dict=False, zero_division=1.0)
        print(results)
