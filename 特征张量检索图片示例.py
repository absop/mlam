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

    def get_paths(self, index):
        return [s[0] for s in self.dataset.samples[index]]

    def __getitem__(self, index):
        if (isinstance(index, slice)):
            return self.get_paths(index), self.features[index].mean(0)

        if (isinstance(index, int)):
            return self.get_paths(index), self.features[index]

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
    prefix = "Retrieval/特征张量检索图片"

    for i, label in enumerate(qf.dataset.classes):
        p, f = qf[label][:8]
        S, P, T = ex.sort_with_similarity(f)

        # Compute dS of S
        dS = np.diff(S)
        _, R = torch.min(torch.Tensor(dS)[(S > 0.6)[:-1]], 0)
        R = R.item() + 1

        iout = imgcat(p, (2, 4), pad=2, w=128)
        iout.save(prefix + label + '示例 输入.png')

        iout = imgcat(P[R - 16:R + 16], (8, 4), pad=2, w=128)
        iout.save(prefix + label + '示例 输出.png')


if __name__ == '__main__':
    main()
