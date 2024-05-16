from pathlib import Path

import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from ssl_pretrain.dinopretrain import DINO

# Load your model and modify it to return patch embeddings
# Example with a dummy model returning random patch embeddings

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


def make_classification_eval_transform(
        *,
        resize_size: int = 256,
        interpolation=transforms.InterpolationMode.BICUBIC,
        crop_size: int = 224,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


class MMORDataset(torch.utils.data.Dataset):
    def __init__(self, transform, take_path):
        self.transform = transform
        self.take_path = take_path
        self.colorimages_path = take_path / 'colorimage'

        self.all_image_paths = []
        for color_image_path in self.colorimages_path.glob('*camera01*.jpg'):
            self.all_image_paths.append(color_image_path)
        for color_image_path in self.colorimages_path.glob('*camera04*.jpg'):
            self.all_image_paths.append(color_image_path)
        for color_image_path in self.colorimages_path.glob('*camera05*.jpg'):
            self.all_image_paths.append(color_image_path)

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        image_path = self.all_image_paths[idx]
        # load image
        image = Image.open(image_path)
        image = self.transform(image)
        target = 0
        image_name = image_path.name
        return image, target, image_name, str(image_path)


def main():
    take_path = Path('/home/guests/ege_oezsoy/MM-OR_processed/035_PKA')
    SCALE = 4
    DINO_SIZE = 'l'

    export_dino_features_path = take_path / f'dino_features_{DINO_SIZE}'
    if not export_dino_features_path.exists():
        export_dino_features_path.mkdir()
    model = DINO(SIZE=DINO_SIZE)  # xFormers==0.0.22
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    # Set up your dataset and dataloader
    # Replace this with your actual dataset
    transform = make_classification_eval_transform(resize_size=256 * SCALE, crop_size=224 * SCALE)

    dataset = MMORDataset(transform=transform, take_path=take_path)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=12, shuffle=False)

    for imgs, targets, names, image_paths in tqdm(dataloader):
        with torch.no_grad():
            # first check if these embeddings exists. If all exists, skip this batch
            to_skip = False
            for name in names:
                embedding_path = export_dino_features_path / f'{name.replace(".jpg", "")}.pt'
                if not embedding_path.exists():
                    to_skip = True
                    break
            if to_skip:
                print(f'Skipping batch...')
                continue
            embeddings = model.get_patch_embeddings(imgs.to(device))
            embeddings = embeddings.reshape(-1, 64, 64, model.teacher_backbone.embed_dim)
            embeddings = embeddings.permute(0, 3, 1, 2)
            embeddings = torch.nn.functional.interpolate(embeddings, size=(8, 8), mode='bilinear', align_corners=False)
            embeddings = embeddings.permute(0, 2, 3, 1)
            if (embeddings * 8).max() > 127 or (embeddings * 8).min() < -128:
                print(f'Over/Under flow detected')
            embeddings = (embeddings * 8).char().cpu()
            for embedding, name, image_path in zip(embeddings, names, image_paths):
                embedding_path = export_dino_features_path / f'{name.replace(".jpg", "")}.pt'
                torch.save(embedding.clone(), embedding_path)


if __name__ == '__main__':
    main()
