import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from LLaVA.llava.mm_utils import get_model_name_from_path, process_images
from LLaVA.llava.model.builder import load_pretrained_model

class ImageDataset(Dataset):
    def __init__(self, img_path, use_aug, num_aug=10):
        self.image_paths = [os.path.join(img_path, fname) for fname in os.listdir(img_path) if fname.endswith('.jpg')]
        self.use_aug = use_aug
        self.num_aug = num_aug
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            # Add more augmentations as desired
        ]) if use_aug else None

    def __len__(self):
        return len(self.image_paths) * self.num_aug if self.use_aug else len(self.image_paths)

    def __getitem__(self, idx):
        if self.use_aug:
            img_idx = idx // self.num_aug
            aug_idx = idx % self.num_aug
            image_path = self.image_paths[img_idx]
            image = Image.open(image_path).convert('RGB')
            augmented_image = self.augmentations(image)
            return augmented_image, os.path.basename(image_path).split('.')[0] + f'_aug_{aug_idx}' + '.pt'
        else:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            return image, os.path.basename(image_path).split('.')[0] + '.pt'

def process_and_extract_features(img_batch, model, image_processor):
    img_batch = process_images(img_batch, image_processor, model.config)
    img_batch = img_batch.to(torch.float16)
    image_features_batch = model.get_vision_tower().global_forward(img_batch.squeeze())
    return model.model.mm_projector(image_features_batch)

if __name__ == '__main__':
    # Load vision encoder of LLaVA
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=None, model_name=model_name, load_8bit=False, load_4bit=False)
    model = model.to(torch.float16)

    USE_AUG = True
    img_path = "data/original_crops"
    img_path = "synthetic_or_generation/vis_descriptors/"
    batch_size = 12 if USE_AUG else 128

    # Create Dataset and DataLoader
    dataset = ImageDataset(img_path, USE_AUG)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch_images, batch_paths in tqdm(dataloader):
        img_batch = process_images(batch_images, image_processor, model.config)

        # Extract features
        img_batch = img_batch.to(torch.float16)
        image_features_batch = process_and_extract_features(img_batch, model, image_processor)

        # Save features for each image in the batch
        for j, image_feature in enumerate(image_features_batch):
            torch.save(image_feature, img_path + batch_paths[j])
