ENTITY_DESCRIPTORS = {
    "head surgeon": [
        "primary operator in surgery, handles critical tasks.",
        "leads surgical team, performs complex procedures.",
        "ovversees and conducts main surgical actions."
    ],
    "assistant surgeon": [
        "supports head surgeon, assists in surgical tasks.",
        "provides necessary tools and assistance to head surgeon.",
        "aids in operative procedures."
    ],
    "circulator": [
        "coordinates OR activities and tools.",
        "manages logistics.",
        "mobilizes around OR, ensures equipment availability.",
    ],
    "nurse": [
        "assists in surgical prep and recovery.",
        "involved in perioperative patient care.",
        "assists surgical team, ensures patient comfort."
    ],
    "anaesthetist": [
        "administers anesthesia, monitors patient.",
        "responsible for patient sedation and pain management.",
        "manages anesthesia delivery, oversees patient's state."
    ],
    "patient": [
        "undergoes surgical procedure.",
        "receives treatment.",
        "subject of surgical operation."
    ],
    "instrument table": [
        "instrument table, blue, large, rectangular, matte."
    ],
    "operating table": [
        "operating table, black, large, rectangular, rubber."
    ],
    "secondary table": [
        "secondary table, gray, large, rectangular, metal."
    ],
    "anesthesia equipment": [
        "anesthesia equipment, white, large, irregular, matte."
    ],
    "instrument": [
        "instrument, handheld."
    ]
}

PREDICATE_DESCRIPTORS = {
    "assisting": [
        "collaboration between staff.",
        "supportive actions by assistant surgeon to aid head surgeon in operative tasks.",
        "help in surgical procedures."
    ],
    "cementing": [
        "use of a tool: black, handheld, straight, metal, surgical bone cement gun.",
    ],
    "cleaning": [
        "use of a tool: white, handheld, irregular, sanitization equipment.",
    ],
    "closeTo": [
        "proximity of medical staff or equipment to each other in OR.",
        "nearness of surgical staff to others.",
        "spatial closeness in OR."
    ],
    "cutting": [
        "use of a tool: white, small, straight, plastic, surgical scalpel.",
    ],
    "drilling": [
        "use of a tool: orange, handheld, L-shape, plastic, surgical drill.",
    ],
    "hammering": [
        "use of a tool: brown, handheld, T-shape, metal, surgical hammer.",
    ],
    "holding": [
        "grasping surgical instruments.",
        "medical staff holding tools.",
        "handling of medical tools."
    ],
    "lyingOn": [
        "patient positioned on the operating table.",
        "patient's placement on operating table.",
        "operative position of the patient on the surgery table."
    ],
    "manipulating": [
        "handling of medical objects like operating tables or anesthesia machines.",
        "adjusting or operating medical equipment by medical staff."
    ],
    "preparing": [
        "includes draping and sterilization.",
        "pre-operative procedures to prepare patient for surgery.",
        "setting up patient and surgical area for the procedure."
    ],
    "sawing": [
        "use of a tool: green, handheld, round, plastic, surgical bone saw.",
    ],
    "suturing": [
        "use of a tool: gray, small, straight, metal, surgical scissors.",
    ],
    "touching": [
        "physical contact between entities.",
        "direct interaction with patient or surgical tools.",
        "tactile engagement in surgical environment."
    ]
}

ENTITY_DESCRIPTORS_TRAINING = {
    "head surgeon": [
        "primary operator in surgery, handles critical tasks.",
        "leads surgical team, performs complex procedures.",
        "oversees and conducts main surgical actions."
    ],
    "assistant surgeon": [
        "supports head surgeon, assists in surgical tasks.",
        "provides necessary tools and assistance to head surgeon.",
        "aids in operative procedures."
    ],
    "circulator": [
        "coordinates OR activities and tools.",
        "manages logistics.",
        "mobilizes around OR, ensures equipment availability.",
    ],
    "nurse": [
        "assists in surgical prep and recovery.",
        "involved in perioperative patient care.",
        "assists surgical team, ensures patient comfort."
    ],
    "anaesthetist": [
        "administers anesthesia, monitors patient.",
        "responsible for patient sedation and pain management.",
        "manages anesthesia delivery, oversees patient's state."
    ],
    "patient": [
        "undergoes surgical procedure.",
        "receives treatment.",
        "subject of surgical operation."
    ],
    "instrument table": [
        "instrument table, blue, large, rectangular, matte."
    ],
    "operating table": [
        "operating table, black, large, rectangular, rubber."
    ],
    "secondary table": [
        "secondary table, gray, large, rectangular, metal."
    ],
    "anesthesia equipment": {'object_type': 'anesthesia equipment', 'color': 'white', 'size': 'large', 'shape': 'irregular', 'texture': 'matte'},
    "instrument": [
        "instrument, handheld."
    ]
}

PREDICATE_DESCRIPTORS_TRAINING = {
    "assisting": [
        "collaboration between staff.",
        "supportive actions by assistant surgeon to aid head surgeon in operative tasks.",
        "help in surgical procedures."
    ],
    "cementing": {'object_type': 'surgical bone cement gun', 'color': 'black', 'size': 'handheld', 'shape': 'straight', 'texture': 'metal'},
    "cleaning": [
        "use of a tool: white, handheld, irregular, sanitization equipment.",
    ],
    "closeTo": [
        "proximity of medical staff or equipment to each other in OR.",
        "nearness of surgical staff to others.",
        "spatial closeness in OR."
    ],
    "cutting": {'object_type': 'surgical scalpel', 'color': 'white', 'size': 'small', 'shape': 'straight', 'texture': 'plastic'},
    "drilling": {'object_type': 'surgical drill', 'color': 'orange', 'size': 'handheld', 'shape': 'L-shape', 'texture': 'plastic'},
    "hammering": {'object_type': 'surgical hammer', 'color': 'brown', 'size': 'handheld', 'shape': 'T-shape', 'texture': 'metal'},
    "holding": [
        "grasping surgical instruments.",
        "medical staff holding tools.",
        "handling of medical tools."
    ],
    "lyingOn": [
        "patient positioned on the operating table.",
        "patient's placement on operating table.",
        "operative position of the patient on the surgery table."
    ],
    "manipulating": [
        "handling of medical objects like operating tables or anesthesia machines.",
        "adjusting or operating medical equipment by medical staff."
    ],
    "preparing": [
        "includes draping and sterilization.",
        "pre-operative procedures to prepare patient for surgery.",
        "setting up patient and surgical area for the procedure."
    ],
    "sawing": {'object_type': 'surgical bone saw', 'color': 'green', 'size': 'handheld', 'shape': 'round', 'texture': 'plastic'},
    "suturing": {'object_type': 'surgical scissors', 'color': 'gray', 'size': 'small', 'shape': 'straight', 'texture': 'metal'},
    "touching": [
        "physical contact between entities.",
        "direct interaction with patient or surgical tools.",
        "tactile engagement in surgical environment."
    ]
}

ENTITY_SYMBOLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
PREDICATE_SYMBOLS = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω']

if __name__ == '__main__':
    entity_name_to_symbol = {}
    entity_symbol_to_descriptor = {}
    predicate_name_to_symbol = {}
    predicate_symbol_to_descriptor = {}
    entity_symbols = ENTITY_SYMBOLS.copy()
    predicate_symbols = PREDICATE_SYMBOLS.copy()
    for entity_name, descriptors in ENTITY_DESCRIPTORS.items():
        entity_name_to_symbol[entity_name] = entity_symbols.pop(0)
        entity_symbol_to_descriptor[entity_name_to_symbol[entity_name]] = descriptors[0]
    for predicate_name, descriptors in PREDICATE_DESCRIPTORS.items():
        predicate_name_to_symbol[predicate_name] = predicate_symbols.pop(0)
        predicate_symbol_to_descriptor[predicate_name_to_symbol[predicate_name]] = descriptors[0]

    entity_symbol_to_descriptor_sorted = sorted(entity_symbol_to_descriptor.items(), key=lambda x: x[0])
    predicate_symbol_to_descriptor_sorted = sorted(predicate_symbol_to_descriptor.items(), key=lambda x: x[0])
    entity_symbols = ", ".join([elem[0] for elem in entity_symbol_to_descriptor_sorted])
    predicate_symbols = ", ".join([elem[0] for elem in predicate_symbol_to_descriptor_sorted])
    human_prompt = f'Entities: [{entity_symbols}]. Predicates: [{predicate_symbols}]. <knowledge_start> '
    for entity_symbol, descriptor in entity_symbol_to_descriptor_sorted:
        human_prompt += f'{entity_symbol}: {descriptor} '
    for predicate_symbol, descriptor in predicate_symbol_to_descriptor_sorted:
        human_prompt += f'{predicate_symbol}: {descriptor} '
    human_prompt += f'<knowledge_end> Given the following scene graph memory representation, generate a scene graph for timepoint T. The output should strictly be a list of triplets, each in the format "entity1,entity2,predicate;". Do not provide a narrative or descriptive text.'
    print(human_prompt)
