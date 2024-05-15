import os
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label, regionprops
from tqdm.contrib.concurrent import process_map  # Import process_map from tqdm


def save_component_of_specific_size(image_path, target_size):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if image has an alpha channel
    if image.shape[2] == 4:
        # Use the alpha channel for component analysis
        alpha_channel = image[:, :, 3]
    else:
        # For images without alpha channel, use the grayscale image
        alpha_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Label connected components
    labeled = label(alpha_channel)

    # Find the component closest to the target size
    component_sizes = [(region.label, region.area) for region in regionprops(labeled)]
    closest_component = min(component_sizes, key=lambda x: abs(x[1] - target_size))

    # Create a mask for the component
    mask = np.zeros_like(alpha_channel)
    mask[labeled == closest_component[0]] = 255

    # Save the mask
    cv2.imwrite("test.png", mask)


def has_multiple_large_components(image_path, size_threshold):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if image has an alpha channel
    if image.shape[2] == 4:
        # Use the alpha channel for component analysis
        alpha_channel = image[:, :, 3]
    else:
        # For images without alpha channel, use the grayscale image
        alpha_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Label connected components
    labeled = label(alpha_channel)

    # Count the number of components larger than the threshold
    num_large_components = sum(region.area > size_threshold for region in regionprops(labeled))

    return num_large_components > 1


def process_image(image_path):
    # filter out images with multiple large components

    SIZE_THRESHOLD = 1000
    trash_images_dir = 'synthetic_or_generation/images_sdxl_wrong'
    corresponding_json = f'synthetic_or_generation/images_sdxl/{Path(image_path).stem}.json'

    if has_multiple_large_components(image_path, SIZE_THRESHOLD):
        os.rename(image_path, f'{trash_images_dir}/{Path(image_path).name}')
        os.rename(corresponding_json, f'{trash_images_dir}/{Path(corresponding_json).name}')


def main():
    images_dir = 'synthetic_or_generation/images_sdxl'
    image_paths = [str(p) for p in Path(images_dir).glob('*.png')]

    # Using process_map from tqdm for multiprocessing with a progress bar
    process_map(process_image, image_paths, max_workers=12)


if __name__ == '__main__':
    main()
