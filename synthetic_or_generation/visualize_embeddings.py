import os

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

if __name__ == '__main__':
    drilling_crops = []
    sawing_crops = []
    cutting_crops = []

    for fname in os.listdir('data/original_crops/crops_for_embs'):
        if not fname.endswith('.pt'):
            continue
        emb = torch.load(os.path.join('synthetic_or_generation/original_crops/crops_for_embs', fname))
        if 'drill' in fname:
            drilling_crops.append(emb)
        elif 'saw' in fname:
            sawing_crops.append(emb)
        elif 'cut' in fname:
            cutting_crops.append(emb)

    # Convert lists to tensors
    drilling_crops = torch.stack(drilling_crops)
    sawing_crops = torch.stack(sawing_crops)
    cutting_crops = torch.stack(cutting_crops)

    # Concatenate all embeddings and labels
    all_embs = torch.cat([drilling_crops, sawing_crops, cutting_crops], dim=0)
    labels = ['drill'] * len(drilling_crops) + ['saw'] * len(sawing_crops) + ['cut'] * len(cutting_crops)

    # Convert to numpy
    all_embs_np = all_embs.detach().cpu().numpy()

    # Apply PCA with 3 components
    pca = PCA(n_components=3)
    embs_3d = pca.fit_transform(all_embs_np)

    # Plotting in 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'drill': 'red', 'saw': 'blue', 'cut': 'green'}

    for emb, label in zip(embs_3d, labels):
        ax.scatter(emb[0], emb[1], emb[2], color=colors[label], label=label, alpha=0.5)

    # Adding legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

    ax.set_title('3D Visualization of Embeddings with PCA')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    plt.show()

    # Apply PCA with 2 components
    pca = PCA(n_components=2)
    embs_2d = pca.fit_transform(all_embs_np)

    plt.figure(figsize=(8, 6))
    for emb, label in zip(embs_2d, labels):
        plt.scatter(emb[0], emb[1], color=colors[label], label=label, alpha=0.5)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    plt.legend(*zip(*unique))

    plt.title('2D Visualization of Embeddings with PCA')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()
