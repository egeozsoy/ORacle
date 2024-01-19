import os

from PIL import Image
from tqdm import tqdm

from LLaVA.llava.mm_utils import get_model_name_from_path, process_images
from LLaVA.llava.model.builder import load_pretrained_model

import torch
from torchvision import transforms


def random_augmentations(image):
    """Apply random augmentations to the image."""
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        # Add more augmentations as desired
    ])
    return augmentations(image)

def process_and_extract_features(img_batch):
    """Process a batch of images and extract features."""
    img_batch = process_images(img_batch, image_processor, model.config)
    img_batch = img_batch.to(torch.float16)
    image_features_batch = model.get_vision_tower().global_forward(img_batch)
    return model.model.mm_projector(image_features_batch)

if __name__ == '__main__':
    # Load vision encoder of LLaVA
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=None, model_name=model_name, load_8bit=False, load_4bit=False)
    # convert model to bfloat16
    model = model.to(torch.float16)

    USE_AUG = False
    # Define batch size
    batch_size = 12 if USE_AUG else 128 # Adjust this based on your memory constraints -> *10 for augmentations

    # List all image paths
    # image_paths = [os.path.join('synthetic_or_generation/vis_descriptors', fname)
    #                for fname in os.listdir('synthetic_or_generation/vis_descriptors')
    #                if fname.endswith('.jpg')]

    # original 4D OR
    image_paths = [os.path.join('data/original_crops', fname) for fname in os.listdir('data/original_crops') if fname.endswith('.jpg')]

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [Image.open(path).convert('RGB') for path in batch_paths]
        batch_augmented_images = []
        batch_augmented_paths = []

        if USE_AUG:
            for image in batch_images:
                for aug_i in range(10):
                    augmented_image = random_augmentations(image)
                    batch_augmented_images.append(augmented_image)
                    batch_augmented_paths.append(os.path.basename(batch_paths[i]).split('.')[0] + f'aug_{aug_i}' + '.pt')
            batch_images = batch_augmented_images
            batch_paths = batch_augmented_paths

        # Process images in batch
        img_batch = process_images(batch_images, image_processor, model.config)

        # Extract features in batch
        img_batch = img_batch.to(torch.float16)
        image_features_batch = model.get_vision_tower().global_forward(img_batch.squeeze())
        image_features_batch = model.model.mm_projector(image_features_batch)

        # Save features for each image in the batch
        for j, image_feature in enumerate(image_features_batch):
            if USE_AUG:
                torch.save(image_feature, 'synthetic_or_generation/vis_descriptors/' + batch_paths[j])
            else:
                torch.save(image_feature, 'data/original_crops/' + os.path.basename(batch_paths[j]).split('.')[0] + '.pt')