# 1) Define Attribute and Descriptors. Color, Size, Shape, Texture, Object Type
# 2) Use stable-diffusion xl with a correct prompt to generate these images.
# 3) Use DIS to remove the background and get pngs. Save each image with its corresponding attributes and descriptors.


import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from tqdm import tqdm

from synthetic_or_generation.dis_background_removal_helpers.data_loader_cache import normalize
from synthetic_or_generation.dis_background_removal_helpers.isnet import ISNetDIS

surgical_tools_and_equipment = [
    {"Object Type": "Surgical Drill", "Function": "Creating holes in bone"},
    {"Object Type": "Surgical Bone Saw", "Function": "Cutting bone"},
    {"Object Type": "Surgical Hammer", "Function": "Fixing implants"},
    {"Object Type": "Surgical Scissors", "Function": "Cutting tissue"},
    {"Object Type": "Surgical Retractor", "Function": "Holding back tissues"},
    {"Object Type": "Surgical Scalpel", "Function": "Incising tissue"},
    {"Object Type": "da Vinci Surgical System", "Function": "Robotic-assisted surgeries"},
    {"Object Type": "Mako Robotic-Arm Assisted Surgery System", "Function": "Orthopedic surgeries with robotic assistance"},
    {"Object Type": "Electrosurgical Unit", "Function": "Cutting and cauterizing tissue"},
    {"Object Type": "Surgical C-arm", "Function": "Intraoperative imaging"},
    {"Object Type": "Microscope for Microsurgery", "Function": "Enhanced visualization for microsurgeries"},
    {"Object Type": "Surgical Forceps", "Function": "Gripping and holding tissues"},
    {"Object Type": "Anesthesia Equipment", "Function": "Delivering anesthesia"},
    {"Object Type": "Surgical Bone Cement Gun", "Function": "Applying bone cement"},
    {"Object Type": "Surgical Navigation System", "Function": "Guidance for precision surgery"},
    {"Object Type": "Surgical Imaging Systems", "Function": "Visualizing internal structures"}
]

# Creating a concise list of 10 most relevant medical staff roles, differentiating between head surgeon and assistant surgeon.

relevant_medical_staff_roles = [
    {"Role": "Head Surgeon", "Function": "Leads surgical procedures and makes critical decisions"},
    {"Role": "Assistant Surgeon", "Function": "Assists the head surgeon, provides surgical support"},
    {"Role": "Anesthesiologist", "Function": "Administers anesthesia, monitors patient vitals"},
    {"Role": "Scrub Nurse", "Function": "Assists in the operating room, handles surgical instruments"},
    {"Role": "Circulating Nurse", "Function": "Manages the overall environment of the operating room"},
    {"Role": "Operating Room Technician", "Function": "Prepares and maintains operating equipment and instruments"},
    {"Role": "Surgical Nurse Practitioner", "Function": "Provides pre- and post-operative care, assists in surgery"},
    {"Role": "Radiologic Technologist", "Function": "Operates imaging equipment during procedures"},
    {"Role": "Anesthesia Technician", "Function": "Assists with anesthesia equipment and medications"},
    {"Role": "Recovery Room Nurse", "Function": "Cares for patients post-surgery in recovery"}
]

ATTRIBUTES = {
    'object_type': [elem['Object Type'].lower() for elem in surgical_tools_and_equipment],
    'color': ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white', 'transparent', 'gray'],
    'size': ['handheld', 'tabletop', 'portable', 'large', 'small'],
    'shape': ['L-shape', 'T-shape', 'straight', 'curved', 'round', 'rectangular', 'irregular', 'flat'],
    'texture': ['plastic', 'metal', 'rubber', 'smooth', 'matte', 'glossy'],
}


def sample_attributes():
    attributes = {}
    for attr, values in ATTRIBUTES.items():
        attributes[attr] = np.random.choice(values)
    return attributes


def main():
    N_GEN = 4
    NUM_INFERENCE_STEPS = 50
    USE_REFINER = True

    version = '1.0'
    base_name = f"stabilityai/stable-diffusion-xl-base-{version}"
    refiner_name = f"stabilityai/stable-diffusion-xl-refiner-{version}"
    dis_model_path = "synthetic_or_generation/dis_background_removal_helpers/saved_models/isnet-general-use.pth"  # the model path
    dis_img_size = [1024, 1024]
    output_dir = Path('synthetic_or_generation/images_sdxl')
    output_dir.mkdir(exist_ok=True)

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # SDXL
    if device == 'cuda':
        pipe = StableDiffusionXLPipeline.from_pretrained(base_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(base_name, use_safetensors=True)
    pipe = pipe.to(device)

    # SDXL Refiner
    if USE_REFINER:
        if device == 'cuda':
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(refiner_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        else:
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(refiner_name, use_safetensors=True)
        refiner = refiner.to(device)

    # DIS (Background Removal)
    dis = ISNetDIS()
    if torch.cuda.is_available():
        dis.load_state_dict(torch.load(dis_model_path))
    else:
        dis.load_state_dict(torch.load(dis_model_path, map_location="cpu"))
    dis = dis.to(device)
    dis.eval()

    # 1) Define Attribute. Color, Size, Shape, Texture, Object Type
    for _ in tqdm(range(10000000)):
        attributes = sample_attributes()
        file_name = f"{attributes['object_type']}_{attributes['color']}_{attributes['size']}_{attributes['shape']}_{attributes['texture']}_{NUM_INFERENCE_STEPS}"
        # Automatically construct the prompt. First we define a prompt that would generally work well for stable diffusion. Then we fill in the blanks.
        prompt = f"A realistic image of a single {attributes['object_type']}, {attributes['color']}, {attributes['size']}, {attributes['shape']}, {attributes['texture']}."

        # Batched image generation
        with torch.no_grad():
            images = pipe([prompt] * N_GEN, output_type="latent" if USE_REFINER else "pil", num_inference_steps=NUM_INFERENCE_STEPS).images
            if USE_REFINER:
                images = refiner(prompt=[prompt] * N_GEN, image=images, num_inference_steps=NUM_INFERENCE_STEPS).images

            # Convert images to tensors for DIS processing
            images = [np.array(im) for im in images]
            im_shapes = [im.shape[0:2] for im in images]
            im_tensors = [F.upsample(torch.tensor(im, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0), dis_img_size, mode="bilinear").type(torch.uint8) for im in images]
            batch_images = torch.cat(im_tensors, dim=0)
            batch_images = torch.divide(batch_images, 255.0)
            batch_images = normalize(batch_images, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

            # Batched DIS background removal
            masks = dis(batch_images.to(device))
            # from here on it should be identical to the old code
            for i, (mask, im_shp, im) in enumerate(zip(masks[0][0], im_shapes, images)):
                mask = torch.squeeze(F.upsample(mask.unsqueeze(0), im_shp, mode='bilinear'), 0)
                ma = torch.max(mask)
                mi = torch.min(mask)
                mask = (mask - mi) / (ma - mi)
                mask = (mask * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
                # this is the mask we apply it to our image
                im = Image.fromarray(im)
                im = im.convert("RGBA")
                mask = Image.fromarray(mask[:, :, 0]).convert('L')
                mask = np.array(mask)
                mask = (mask > 128) * 255  # Apply threshold
                mask = Image.fromarray(mask.astype(np.uint8))
                im.putalpha(mask)
                # Now crop the image to the object
                im = im.crop(im.getbbox())
                im.save(output_dir / f'{file_name}_{i}.png')
                # save the attributes and descriptors as json
                with open(output_dir / f'{file_name}_{i}.json', 'w') as f:
                    json.dump(attributes, f)

            # Clean up everything in preparing for the next batch
            del batch_images
            del masks
            del images
            del im_tensors
            del im_shapes
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
