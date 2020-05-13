import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale
from torchvision import datasets, transforms

import torch
from utils.tsne import load_tsnes

import matplotlib as mpl

mpl.style.use('default')

# plt.rc('text', usetex=True)
plt.rc('font', family='Courier New')


class ImageTSNE(object):
    def __init__(self, root, subdirs):
        self.root = root
        self.subdirs = subdirs
        self.fig = None

    def savefig(self, filename):
        if self.fig == None:
            self.fig = plt.figure()
            ax2d = self.fig.add_subplot(211)
            ax3d = self.fig.add_subplot(212, projection='3d')
            colors = ['cornflowerblue', 'mediumpurple']
            shapes, labels = [], []
            for subdir, rc in self.subdirs.items():
                tsne2d, tsne3d, targets, classes = load_tsnes(
                    "{}/{}".format(self.root, subdir))

                tsne2d = minmax_scale(tsne2d, axis=0)
                tsne3d = minmax_scale(tsne3d, axis=0)
                for i, c in enumerate(classes):
                    labels.append("{} {}".format(subdir, c))
                    sieve = targets == i
                    t2 = tsne2d[sieve]
                    t3 = tsne3d[sieve]
                    # 2D scatter
                    shapes.append(ax2d.scatter(t2[:, 0], t2[:, 1],
                        c=colors[i], **rc))
                    # 3D scatter
                    ax3d.scatter(t3[:, 0], t3[:, 1], t3[:, 2],
                        c=colors[i], **rc)

            lgds = ax3d.legend([shapes[0], shapes[2]], [labels[0], labels[2]],
                fontsize=10, bbox_to_anchor=(1.0, 1.2))
            ax3d.legend([shapes[1], shapes[3]], [labels[1], labels[3]],
                fontsize=10, bbox_to_anchor=(0.35, 1.2))
            ax3d.add_artist(lgds)

            ax2d.set_title('2D t-SNE')
            ax3d.set_title('3D t-SNE')

            ax2d.set_xticks([])
            ax2d.set_yticks([])

            ax3d.set_xticks([])
            ax3d.set_yticks([])
            ax3d.set_zticks([])

        self.fig.savefig(filename, dpi=300)

    def show(self):
        if self.fig is not None:
            plt.show()


"""对 train 和 valid 两个文件夹下的 movie2 和 movie6 两类图片进行TSNE，
   然后显示在一张散点图上， movie2 和 movie6 异色，train 和 valid 异形
"""
def main():
    it = ImageTSNE('images', {
            'train': dict(marker='^', alpha=0.4, edgecolor='k'),
            'valid': dict(marker='o', alpha=0.7, edgecolor='k')
        })
    it.savefig('TSNE/TSNE可视化.png')
    it.savefig('TSNE/TSNE可视化.pdf')
    it.show()


if __name__ == '__main__':
    main()
