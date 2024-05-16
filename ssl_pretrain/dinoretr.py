import random

import torch
from sklearn.decomposition import PCA
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl

from ssl_pretrain.dinopretrain import PretrainDataset, DINO
from scipy.spatial import distance

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


def find_closest_embeddings(query_embedding, all_embeddings, all_names, num_closest=5):
    # Compute the distance between the query and all embeddings
    distances = [distance.euclidean(query_embedding, emb) for emb in all_embeddings]
    # Get the indices of the closest embeddings
    closest_indices = np.argsort(distances)[:num_closest]
    return [all_names[idx] for idx in closest_indices]


def main():
    # model = DINO.load_from_checkpoint(checkpoint_path)
    scale = 1  # TODO try with 1
    model = DINO(SIZE='g')  # xFormers==0.0.22
    # model = DINO.load_from_checkpoint('/home/guests/ege_oezsoy/Oracle/ssl_pretrain/checkpoints/dino_s_g_bs_4_lr_1e-05_100epochs-07-0.15.ckpt', SIZE='g', LR=1e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Set up your dataset and dataloader
    # Replace this with your actual dataset
    transform = make_classification_eval_transform(resize_size=256 * scale, crop_size=224 * scale)
    # fix seed to fix shuffle etc.
    pl.seed_everything(42)
    dataset = PretrainDataset(transform=transform)
    pl.seed_everything(42)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=12, shuffle=True)

    # Compute embeddings for the entire dataset
    all_embeddings = []
    all_paths = []
    count = 5000
    for img, target, name, image_path in tqdm(dataloader, desc='Computing embeddings'):
        with torch.no_grad():
            embedding = model.get_global_embeddings(img.to(device)).cpu()
            all_embeddings.append(embedding)
            all_paths.append(image_path)
            count -= 1
            if count == 0:
                break

    # Flatten the embeddings for easier comparison
    all_embeddings = [emb.view(-1) for emb in all_embeddings]

    # Choose 10 random images from the dataset
    random_indices = random.sample(range(len(dataset)), 20)
    query_images = [(dataset[idx][0], dataset[idx][3]) for idx in random_indices]

    # For each query image, find the 5 closest images
    for idx, (query_img, query_path) in enumerate(query_images):
        query_embedding = model.get_global_embeddings(query_img.unsqueeze(0).to(device)).cpu().view(-1)
        closest_paths = find_closest_embeddings(query_embedding, all_embeddings, all_paths)

        # Visualize the query image and its closest images
        fig, axs = plt.subplots(1, 6, figsize=(15, 5))
        axs[0].imshow(Image.open(query_path))
        axs[0].set_title('Query Image')
        axs[0].axis('off')

        for i, name in enumerate(closest_paths, 1):
            closest_img = Image.open(name[0])  # Assuming the name includes the path
            axs[i].imshow(closest_img)
            axs[i].set_title(f'Closest {i}')
            axs[i].axis('off')

        plt.savefig(f'ssl_pretrain/frozengclosest_images_{scale}_{idx}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)


if __name__ == '__main__':
    main()
