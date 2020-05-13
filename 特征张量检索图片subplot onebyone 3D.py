import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.features import GalleryLoader, load_gallery_features
from utils.features import QueriesLoader, load_queries_features
from utils.features import pretrained_vgg16
from utils.icat import imgcat

import matplotlib as mpl

mpl.style.use('seaborn')

plt.rc('font', family='serif')
Kaiti = {'family': 'Kaiti'}


class QueriesFeatures(QueriesLoader):
    def __init__(self, model, root, label=None, shuffle=True, batch_size=8):
        super(QueriesFeatures, self).__init__(root,
            label=label, shuffle=shuffle, batch_size=batch_size)
        self.model = model
        self.root = root
        self.cache = {}
        if label is not None:
            self.focus_label(label)

    def focus_label(self, label):
        self.dataset.set_label(label)
        if label not in self.cache:
            self.cache[label] = load_queries_features(self.model,
                self.root, label, self)

        self.label = label
        self.features = self.cache[label]

    def __getitem__(self, index):
        if (isinstance(index, slice)):
            return self.features[index].mean(0)

        if (isinstance(index, int)):
            return self.features[index]

        if (isinstance(index, str)):
            self.focus_label(index)
            return self


class ImagesExtracter(GalleryLoader):
    def __init__(self, model, gallery_root):
        super(ImagesExtracter, self).__init__(gallery_root)
        self.gallery_features = load_gallery_features(
            model, gallery_root, self)

    def sort_similarities(self, queries_features):
        similarities, indexes = F.cosine_similarity(self.gallery_features,
            queries_features.view(1, -1)).sort(dim=0, descending=True)
        similarities = np.around(similarities.numpy(), 3)
        return similarities, indexes

    def sort_with_similarity(self, queries_features):
        similarities, indexes = self.sort_similarities(queries_features)
        paths, targets = [], []
        for i in range(len(similarities)):
            path, target = self.dataset.samples[indexes[i]]
            paths.append(path)
            targets.append(target)

        return similarities, paths, np.array(targets)


def main():
    model = pretrained_vgg16()

    # valid(80%) as gallery, and train(20%) as queries
    ex = ImagesExtracter(model, "images/valid")
    qf = QueriesFeatures(model, "images/train")
    ones = np.ones(len(ex.gallery_features))
    x = np.arange(1, 1 + len(ex.gallery_features))
    plt.figure(figsize=(8, 3.5))
    for i, label in enumerate(qf.dataset.classes):
        ax3 = plt.subplot(121 + i, projection='3d')
        for j, f in enumerate(qf[label].features, 1):
            y = j * ones
            z, _, t = ex.sort_with_similarity(f)
            for idx, c in enumerate(ex.dataset.classes):
                sieve = t == idx
                ax3.plot3D(x[sieve], y[sieve], z[sieve],
                    '.', ms=4, color=['C1', 'C2'][idx], label=c)
                # ax3.scatter(x[sieve], y[sieve], z[sieve],
                #     marker='.', s=2, c=colors[idx], label=c)

        ax3.set_title(['(a)', '(b)'][i] + '. %s' % label, y=1.1)
        ax3.set_xlabel('R')
        ax3.set_ylabel('N')
        ax3.set_zlabel('S')

    plt.tight_layout(w_pad=3, pad=3)
    plt.savefig("Retrieval/特征张量检索图片onebyone.png", dpi=300)
    plt.savefig("Retrieval/特征张量检索图片onebyone.pdf", dpi=300)

if __name__ == '__main__':
    main()
