# 1) Surgical tool / equipment generation. Use stable difffusion xl + DIS background removal. Stick to predefined attributes and descriptors
# 2) Use generate_novel_augmentations.py script from novel_or_synthesis project as well as the removal function to automatically create new scenes. For every scene, create purposeful augmentations which can be described correctly.
# 3) Generate new training data for llava. It should sample both from original 4D-OR as well as this synthetic data. The descriptors should generally match the corresponding images. They can sometimes be incomplete, sometimes wrong or synonmy.
# The goal is to teach the model more variability and to be able to generalize better to new surgeries and ORs.
