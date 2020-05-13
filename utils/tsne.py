import os
import numpy as np
import torch

from sklearn.manifold import TSNE
from torchvision import datasets, transforms


tsne2d_function = TSNE(n_components=2, init='pca', random_state=0)
tsne3d_function = TSNE(n_components=3, init='pca', random_state=0)


def load_state_dict(root):
    filepath = os.path.join(root, 'tsnes.pth.tar')
    try:
        state_dict = torch.load(filepath)
    except:
        state_dict = {}

    return filepath, state_dict


def load_tsne2d(root):
    filepath, state_dict = load_state_dict(root)
    try:
        paths = state_dict['paths']
        tsne2d = state_dict['tsne2d']
        labels = state_dict['labels']
        classes = state_dict['classes']
    except:
        paths, images, labels, classes = load_images(root)
        tsne2d = tsne2d_function.fit_transform(images)

        state_dict.update({
            'paths': paths,
            'tsne2d': tsne2d,
            'labels': labels,
            'classes': classes
        })
        torch.save(state_dict, filepath)

    return paths, tsne2d, labels, classes


def load_tsne3d(root):
    filepath, state_dict = load_state_dict(root)
    try:
        paths = state_dict['paths']
        tsne3d = state_dict['tsne3d']
        labels = state_dict['labels']
        classes = state_dict['classes']
    except:
        paths, images, labels, classes = load_images(root)
        tsne3d = tsne3d_function.fit_transform(images)

        state_dict.update({
            'paths': paths,
            'tsne3d': tsne3d,
            'labels': labels,
            'classes': classes
        })
        torch.save(state_dict, filepath)

    return paths, tsne3d, labels, classes


def load_images(root):
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(),
        # transforms.ToTensor(),
        np.array
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=root, transform=data_transform)
    paths = [im[0] for im in dataset.imgs]

    images = np.array([im[0] for im in dataset])
    labels = np.array([im[1] for im in dataset])

    images = images.reshape(images.shape[0], -1)
    classes = dataset.classes

    return paths, images, labels, classes
