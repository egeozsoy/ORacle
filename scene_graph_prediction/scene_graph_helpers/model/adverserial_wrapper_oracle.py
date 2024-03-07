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

    def _symbolic_prompt_maker(self, customizations, visual_prompt_classes=None):
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

        # TODO do all the modifications here
        # TODO it is actually as simple as adding any new entities with corresponding descriptions, or overwriting existing ones. Same for predicates. The rest is automatically handled.
        for name, descriptor, type in customizations:
            if type == 'e':
                entity_descriptors[name] = [descriptor]
            elif type == 'p':
                predicate_descriptors[name] = [descriptor]
            else:
                raise NotImplementedError()

        for entity_name, descriptors in entity_descriptors.items():
            entity_name_to_symbol[entity_name] = entity_symbols.pop(0)
            entity_symbol_to_descriptor[entity_name_to_symbol[entity_name]] = descriptors[0]
        for predicate_name, descriptors in predicate_descriptors.items():
            predicate_name_to_symbol[predicate_name] = predicate_symbols.pop(0)
            predicate_symbol_to_descriptor[predicate_name_to_symbol[predicate_name]] = descriptors[0]

        entity_symbol_to_descriptor_sorted = sorted(entity_symbol_to_descriptor.items(), key=lambda x: x[0])
        predicate_symbol_to_descriptor_sorted = sorted(predicate_symbol_to_descriptor.items(), key=lambda x: x[0])
        entity_symbols = ", ".join([elem[0] for elem in entity_symbol_to_descriptor_sorted])
        predicate_symbols = ", ".join([elem[0] for elem in predicate_symbol_to_descriptor_sorted])
        human_prompt = f'Entities: [{entity_symbols}]. Predicates: [{predicate_symbols}]. <knowledge_start> '
        # TODO we can check visual_prompt_classes list again and decide if we want to use the textual prompt or the visual prompt
        for entity_symbol, descriptor in entity_symbol_to_descriptor_sorted:
            human_prompt += f'{entity_symbol}: {descriptor} '
        for predicate_symbol, descriptor in predicate_symbol_to_descriptor_sorted:
            human_prompt += f'{predicate_symbol}: {descriptor} '
        human_prompt += f'<knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'

        # inp = f'Entities: [A, B, C, D, E, F, G, H, I, J, K]. Predicates: [α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ]. <knowledge_start> A: primary operator in surgery, handles critical tasks. B: supports head surgeon, assists in surgical tasks. C: coordinates OR activities and tools. D: assists in surgical prep and recovery. E: administers anesthesia, monitors patient. F: undergoes surgical procedure. G: instrument table, blue, large, rectangular, matte. H: operating table, black, large, rectangular, rubber. I: secondary table, gray, large, rectangular, metal. J: {VIS_DESCRIPTOR_TOKEN}. K: Instrument, handheld. α: collaboration between staff. β: use of a tool: {VIS_DESCRIPTOR_TOKEN}. γ: use of a tool: white, handheld, irregular, sanitization equipment. δ: proximity of medical staff or equipment to each other in OR. ε: use of a tool: {VIS_DESCRIPTOR_TOKEN}. ζ: use of a tool: {VIS_DESCRIPTOR_TOKEN}. η: use of a tool: {VIS_DESCRIPTOR_TOKEN}. θ: grasping surgical instruments. ι: patient positioned on the operating table. κ: handling of medical objects like operating tables or anesthesia machines. λ: includes draping and sterilization. μ: use of a tool: {VIS_DESCRIPTOR_TOKEN}. ν: use of a tool: {VIS_DESCRIPTOR_TOKEN}. ξ: physical contact between entities. <knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
        return human_prompt, entity_name_to_symbol, predicate_name_to_symbol

    def forward(self, batch, ):
        assert len(batch) == 1
        elem = batch[0]
        image_path = elem['image_path']
        test_condition = elem['test_condition']
        vis_desc_path = elem[
            'vis_desc_path']  # TODO this would need to be changed, it is only representing the new path. Somehow all the other features should be gathered here, and this feature should be placed exactly in the correct location in the prompt.
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
            if label == 'hammering':
                customizations = [('hammering', f'use of a tool: {VIS_DESCRIPTOR_TOKEN}', 'p')]
            else:
                raise NotImplementedError()
            # TODO rest

            inp, entity_name_to_symbol, predicate_name_to_symbol = self._symbolic_prompt_maker(customizations, visual_prompt_classes=[])  # TODO actually support thsi
            # order of img_patches: anesthesia machine, cementing, cutting, drilling, hammering, sawing, suturing
            vis_desc = [des.to(self.model.device, dtype=torch.bfloat16) for des in elem['vis_descriptor_embs']]  # TODO we need a better way to guarantee the order of these
        elif '_symbolic' in self.model_name:
            textual_str = f'{textual_attributes["color"]}, {textual_attributes["size"]}, {textual_attributes["shape"]}, {textual_attributes["texture"]}, {textual_attributes["object_type"]}.'
            if label in ['cementing', 'cutting', 'drilling', 'hammering', 'robotic_sawing', 'sawing', 'suturing']:
                if label == 'robotic_sawing':
                    label = 'sawing'
                customizations = [(label, f'use of a tool: {textual_str}', 'p')]
            elif label in ['kuka', 'mako']:
                customizations = [(label, f'{textual_str}', 'e')]
            else:
                raise NotImplementedError(label)
            inp, entity_name_to_symbol, predicate_name_to_symbol = self._symbolic_prompt_maker(customizations)
        else:
            inp = 'Entities: [head surgeon, assistant surgeon, circulator, nurse, anaesthetist, patient, instrument table, operating table, secondary table, anesthesia equipment, instrument]. Predicates: [assisting, cementing, cleaning, closeTo, cutting, drilling, hammering, holding, lyingOn, manipulating, preparing, sawing, suturing, touching]. Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'

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
            # print(f'Length of input_ids: {input_ids.shape[1]}')
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                use_cache=True,
                max_new_tokens=300,
                stopping_criteria=[stopping_criteria],
                vis_descriptor_embs=[vis_desc] if 'USE_VIS_DESC' in self.config and self.config['USE_VIS_DESC'] else None
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
