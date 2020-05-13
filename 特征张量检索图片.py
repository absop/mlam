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

import matplotlib

matplotlib.style.use('seaborn')

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
    colors = ['C1', 'C2']

    prefix = "Retrieval/特征张量检索图片"
    for i, label in enumerate(qf.dataset.classes):
        f = qf[label][7]
        S, P, T = ex.sort_with_similarity(f)
        dS = np.diff(S)
        _, R = torch.min(torch.Tensor(dS)[(S > 0.6)[:-1]], 0)
        R = R.item() + 1
        M = sum(T[:R] == qf.dataset.target)
        A = 100 * M / R

        plt.close('all')
        plt.figure()

        for idx, c in enumerate(ex.dataset.classes):
            sieve = T == idx
            plt.plot(x[sieve], S[sieve],
                '.', ms=4, color=colors[idx], label=c)
            # plt.scatter(x[sieve], S[sieve],
            #     marker='.', s=2, c=colors[idx], label=c)
        plt.plot(np.arange(2, 2 + len(dS)), dS, label='dS')

        x_text = R + 5
        ylim = plt.gca().axes.get_ylim()
        plt.vlines(R, *ylim, color="purple", linewidth=0.5, label=u'分界线')
        plt.text(x_text, 0.15, "$%d$" % R)
        plt.text(x_text, 0.05, "$%.2f\\%%$" % A)
        plt.legend(prop=Kaiti)

        # plt.title(u'检索' + label, Kaiti)
        plt.xlabel('R')
        plt.ylabel('S')


        plt.savefig(prefix + label + '.png', dpi=300)
        plt.savefig(prefix + label + '.pdf', dpi=300)

        iout = imgcat(P[R - 20:R + 20], (10, 4), pad=2, w=128)
        iout.save(prefix + label + '第%d~%d张.png' % (R - 50, R + 50))


        plt.close('all')
        ax3 = plt.axes(projection='3d')

        for n, f in enumerate(qf[label].features, 1):
            y = n * ones
            z, _, t = ex.sort_with_similarity(f)
            # ax.scatter3D(y, x, z, c=np.array(t) * 100, marker='.')
            for idx, c in enumerate(ex.dataset.classes):
                sieve = t == idx
                ax3.plot3D(x[sieve], y[sieve], z[sieve],
                    '.', ms=4, color=colors[idx], label=c)
                # ax3.scatter(x[sieve], y[sieve], z[sieve],
                #     marker='.', s=2, c=colors[idx], label=c)

        # plt.title(u'检索 ' + label, Kaiti)
        ax3.set_xlabel('R')
        ax3.set_ylabel('N')
        ax3.set_zlabel('S')

        plt.savefig(prefix + label + ' ones' + '.png', dpi=300)
        plt.savefig(prefix + label + ' ones' + '.pdf', dpi=300)

if __name__ == '__main__':
    main()
