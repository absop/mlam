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

import matplotlib as mpl

mpl.style.use('seaborn')
# mpl.use("pgf")
# mpl.rcParams.update({
#     "pgf.texsystem": "xelatex",
#     "pgf.preamble": [
#         r"\usepackage[utf8x]{inputenc}",
#         r"\usepackage[T1]{fontenc}",
#         r"\usepackage{cmbright}",
#     ],
#     'axes.unicode_minus': False,
#     'text.usetex': True,
#     'font.family': 'serif',
#     'font.serif': ['Palatino', 'Helvetica'] + mpl.rcParams['font.serif'],
#     "font.sans-serif": ['Kaiti'] + mpl.rcParams['font.sans-serif']
# })
plt.rc('font', family='serif')
Lucida = {'family': 'Lucida Console'}
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


def draw(ex, qf, _dict, n_image=None):
    ones = np.ones(len(ex.gallery_features))
    x = np.arange(1, 1 + len(ex.gallery_features))
    colors = ['C1', 'C2']
    plt.close('all')
    plt.figure(figsize=(8, 4.))
    for i, label in enumerate(qf.dataset.classes):
        f = qf[label][:n_image or len(qf[label].features)]
        S, P, T = ex.sort_with_similarity(f)

        # Compute dS of S
        dS = np.diff(S)
        _, Nr = torch.min(torch.Tensor(dS)[S[:-1] > 0.6], 0)
        Nt = sum(T == qf.dataset.target)
        Nr = Nr.item() + 1
        Nc = sum(T[:Nr] == qf.dataset.target)
        Ne = Nr - Nc
        Nm = Nt - Nc

        ax2 = plt.subplot(121 + i)

        for idx, c in enumerate(ex.dataset.classes):
            sieve = T == idx
            ax2.plot(x[sieve], S[sieve],
                '.', ms=4, color=colors[idx], label=c)
            # ax2.scatter(x[sieve], S[sieve],
            #     marker='.', s=2, c=colors[idx], label=c)
        ax2.plot(np.arange(2, 2 + len(dS)), dS, linewidth=1.0, label='dS')

        if label not in _dict:
            _dict[label] = {'Nt': Nt, 'Ni': [], 'Nr': [], 'Nc': [], 'Ne': [], 'Nm': []}

        _dict[label]['Ni'].append('%s' % str(n_image or 'all training images'))
        _dict[label]['Nr'].append('%d(%.2f%%)' % (Nr, 100.0 * Nr/Nt))
        _dict[label]['Nc'].append('%d(%.2f%%)' % (Nc, 100.0 * Nc/Nr))
        _dict[label]['Ne'].append('%d(%.2f%%)' % (Ne, 100.0 * Ne/Nr))
        _dict[label]['Nm'].append('%d(%.2f%%)' % (Nm, 100.0 * Nm/Nt))

        x_text = Nr + 5
        ylim = ax2.axes.get_ylim()
        ax2.vlines(Nr, *ylim, linewidth=0.5, color='purple', label='分界线')
        ax2.text(x_text, 0.40, 'Nt: %d' % Nt, Lucida)  # 总数
        ax2.text(x_text, 0.34, 'Nr: %d(%.2f%%)' % (Nr, 100.0 * Nr/Nt), Lucida) # 检出数，检出率
        ax2.text(x_text, 0.28, 'Nc: %d(%.2f%%)' % (Nc, 100.0 * Nc/Nr), Lucida) # 检出正确数，检出正确率
        ax2.text(x_text, 0.22, 'Ne: %d(%.2f%%)' % (Ne, 100.0 * Ne/Nr), Lucida) # 检出错误数，检出错误率
        ax2.text(x_text, 0.16, 'Nm: %d(%.2f%%)' % (Nm, 100.0 * Nm/Nt), Lucida) # 未检出，未检出率
        ax2.legend(prop=Kaiti)


        # ax2.set_title(u'检索' + label, Kaiti)
        ax2.set_title(['(a)', '(b)'][i])
        ax2.set_xlabel("R")
        ax2.set_ylabel("S")

    plt.tight_layout(w_pad=0.2, h_pad=0.5, pad=3)
    plt.savefig("Retrieval/特征张量检索图片示例/%s.png" % str(n_image or 'all'),
        bbox_inches='tight', dpi=300)
    plt.savefig("Retrieval/特征张量检索图片示例/%s.pdf" % str(n_image or 'all'),
        bbox_inches='tight', dpi=300)


def main():
    model = pretrained_vgg16()

    # valid(80%) as gallery, and train(20%) as queries
    ex = ImagesExtracter(model, "images/valid")
    qf = QueriesFeatures(model, "images/train")
    _dict = {}

    # draw(ex, qf, _dict, 1)
    # draw(ex, qf, _dict, 2)
    # draw(ex, qf, _dict, 3)
    # draw(ex, qf, _dict, 4)
    # draw(ex, qf, _dict, 5)
    # draw(ex, qf, _dict, 10)
    # draw(ex, qf, _dict, 15)
    # draw(ex, qf, _dict, 20)
    # draw(ex, qf, _dict, 25)
    # draw(ex, qf, _dict)
    draw(ex, qf, _dict, 8)

    # table = ""
    # headers = ['Ni', 'Nr', 'Nc', 'Ne', 'Nm']
    # for label in _dict:
    #     Nt = _dict[label].pop('Nt')
    #     table += ' & '.join(headers) + '\\\\\n'

    #     for i in range(len(_dict[label]['Ni'])):
    #         items = [_dict[label][h][i] for h in headers]
    #         table += ' & '.join(items) + '\\\\\n'

    #     with open('Retrieval/特征张量检索图片示例/%s.txt' % label, 'wb+') as f:
    #         f.write(table)


if __name__ == '__main__':
    main()
