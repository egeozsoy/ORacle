import torch
from sklearn.decomposition import PCA
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from ssl_pretrain.dinopretrain import PretrainDataset, DINO

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


def main():
    # model = DINO.load_from_checkpoint(checkpoint_path)
    scale = 4
    # model = DINO()  # xFormers==0.0.22
    model = DINO.load_from_checkpoint('/home/guests/ege_oezsoy/Oracle/ssl_pretrain/checkpoints/dino_s_l_bs_32_lr_1e-05_100epochs-51-0.04.ckpt', SIZE='l', LR=1e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Set up your dataset and dataloader
    # Replace this with your actual dataset
    transform = make_classification_eval_transform(resize_size=256 * scale, crop_size=224 * scale)
    dataset = PretrainDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    # Extract patch embeddings
    embeddings = []
    # with torch.no_grad():
    #     embedding = model.get_patch_embeddings(transform(Image.open('ssl_pretrain/labrador.jpg')).unsqueeze(0).to(device)).cpu()
    #     embeddings.append(embedding)
    count = 10
    for img, target, name in dataloader:
        with torch.no_grad():
            embedding = model.get_patch_embeddings(img.to(device)).cpu()
            embeddings.append(embedding)

        count -= 1
        if count == 0:
            break

    embeddings = torch.cat(embeddings)

    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(embeddings.view(-1, embeddings.size(-1)))
    pca_result = pca_result.reshape(-1, 16 * scale, 16 * scale, 3)  # Adjust the shape to match your patches
    threshold = pca_result[:, :, :, 0] < 0
    # make sure it is between 0 and 1
    pca_result = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    # Post-process to remove background
    pca_result[threshold] = 0  # Assuming negative values on first component are background

    # Visualize the first three PCA components
    fig, axs = plt.subplots(1, len(pca_result), figsize=(20, 10))
    # Remove padding and margins around the images
    for i, ax in enumerate(axs.flat):
        ax.imshow(pca_result[i])
        ax.set_axis_off()  # Hide the axes

    plt.subplots_adjust(wspace=0, hspace=0)  # Adjust the space between images
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # No x-axis ticks
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # No y-axis ticks

    # Save the figure
    plt.savefig(f'ssl_pretrain/pca_result_{scale}.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()
