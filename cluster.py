import torch
from torchvision import datasets, transforms

import os
import json
import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from utils.features import load_gallery_features_vgg16

import matplotlib as mpl

mpl.style.use('seaborn')
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def predict(data, max_cluster):
    SSEs = np.ndarray(max_cluster)

    for k in range(1, 1 + max_cluster):
        est = KMeans(n_clusters=k)
        est.fit(data)
        SSEs[k - 1] = est.inertia_

    _, cluster = torch.min(torch.Tensor(np.diff(SSEs)), 0)

    cluster += 2

    return SSEs, cluster


def load_predict(root, max_cluster=8):
    filepath = os.path.join(root, 'Cluster-SSEs.json')
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            return [(np.array(p['SSEs']), p['cluster']) for p in data]

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(),
        # transforms.ToTensor(),
        np.array,
        # lambda x: x / 255.0
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    features = load_gallery_features_vgg16(root)
    dataset = datasets.ImageFolder(root=root, transform=data_transform)
    images = np.array([im[0] for im in dataset])
    images = images.reshape(images.shape[0], -1)

    iSSEs, icluster = predict(images, max_cluster)
    fSSEs, fcluster = predict(features, max_cluster)

    with open(filepath, 'w+') as f:
        json.dump([
            {'SSEs': list(iSSEs), 'cluster': int(icluster)},
            {'SSEs': list(fSSEs), 'cluster': int(fcluster)}
        ], f)

    return [(iSSEs, icluster), (fSSEs, fcluster)]


def main():
    max_cluster = 8
    results = load_predict('images', max_cluster)
    plt.figure(figsize=(8, 5.0))
    for i in range(2):
        SSEs, cluster = results[i]
        plt.subplot(121 + i)
        plt.plot(np.arange(1, 1 + max_cluster), SSEs)
        plt.annotate('%d cluster' % results[i][1],
            xy=(cluster, results[i][0][cluster - 1]),
            xytext=(cluster + 1, SSEs[cluster - 1] * 1.1),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.title(['images', 'features'][i])
        plt.xlabel('k')
        plt.ylabel('SSE')

    plt.tight_layout(w_pad=1.2)
    # plt.suptitle('SSE of k-means cluster')

    plt.savefig("Cluster/cluster-SSE.png", dpi=300)
    plt.savefig("Cluster/cluster-SSE.pdf", dpi=300)
    # plt.show()


if __name__ == '__main__':
    main()
