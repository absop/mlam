import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from PIL import Image
from matplotlib import pyplot as plt

import matplotlib as mpl

mpl.style.use('seaborn')

plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

Kaiti = {'family': 'Kaiti'}

data_transform = transforms.Compose([
    transforms.Resize(192),
    transforms.CenterCrop(192),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # lambda x: x / 255.0
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

class RetrievalImages(datasets.ImageFolder):
    def __init__(self, root, label=None, transform=data_transform):
        super(RetrievalImages, self).__init__(root, transform=transform)
        samples = [torch.Tensor() for i in self.classes]
        for path, t in self.samples:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)

            samples[t] = torch.cat((samples[t], sample.view(1, -1)), 0)

        self.samples = samples
        self.root = root

        if label is not None:
            self.set_label(label)

    def set_label(self, label):
        self.target = self.class_to_idx[label]
        self.images = self.samples[self.target]

    def __getitem__(self, index):
        if (isinstance(index, slice)):
            return self.images[index].mean(0)

        if (isinstance(index, int)):
            return self.images[index]

        if (isinstance(index, str)):
            self.set_label(index)
            return self


class ImagesExtracter(datasets.ImageFolder):
    def __init__(self, gallery_root):
        super(ImagesExtracter, self).__init__(gallery_root,
            transform=data_transform)

        images = torch.Tensor()
        for path, t in self.samples:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)

            images = torch.cat((images, sample.view(1, -1)), 0)

        self.images = images

    def sort_with_distance(self, image):
        distances = ((self.images - image)**2).sum(1).sqrt()
        distances, indexes = distances.sort(dim=0, descending=False)

        distances = np.around(distances.numpy(), 3)

        paths, targets = [], []
        for i in range(len(distances)):
            path, target = self.samples[indexes[i]]
            paths.append(path)
            targets.append(target)

        return distances, paths, np.array(targets)


def draw1(ext, rti):
    plt.close('all')
    plt.figure(figsize=(8, 5))
    x = np.arange(1, 1 + len(ext.images))
    colors = ['C1', 'C2']
    for i, label in enumerate(rti.classes):
        image = rti[label][:5]
        D, P, T = ext.sort_with_distance(image)

        # Compute differences of distances
        dD = np.diff(D)
        _, R = torch.max(torch.Tensor(dD), 0)

        R = R.item() + 1
        A = 100.0 * sum(T[:R] == rti.target) / R

        plt.subplot(121 + i)
        for t, c in enumerate(rti.classes):
            sieve = T == t
            plt.plot(x[sieve], D[sieve], '.', ms=4, color=colors[t], label=c)
        plt.plot(np.arange(2, 2 + len(dD)), dD, label="d(dist)")

        ylim = plt.gca().get_ylim()
        plt.vlines(R, *ylim, color="purple", linewidth=0.5, label=u'分界线')

        x_text = 0

        plt.title(label)
        plt.xlabel("R") # Rank
        plt.ylabel("dist") # Similarity
        plt.legend(prop=Kaiti)

        plt.text(R + 5, 30, '$%d$' % R)
        plt.text(R + 5, 25, '$%.2f\\%%$' % A)

    plt.tight_layout(w_pad=1.2)
    # plt.suptitle(u'直接欧氏距离检索图片', **Kaiti)
    plt.savefig(u'Retrieval/直接欧式距离检索图片.png', dpi=300)
    plt.savefig(u'Retrieval/直接欧式距离检索图片.pdf', dpi=300)


def draw2(ext, rti):
    plt.close('all')
    plt.figure()
    for i, label in enumerate(rti.classes):
        plt.subplot(121 + i)
        Rns = []
        Ans = []
        Accs = []
        for image in rti[label].images[::1]:
            image = image.mean(0)
            D, paths, T = ext.sort_with_distance(image)

            # Compute differences of distances
            Dd = np.diff(D)
            _, Rn = torch.max(torch.Tensor(Dd)[(D > 0.5)[:-1]], 0)

            Rn = Rn.item() + 1
            An = sum(T[:Rn] == rti.target)
            Acc = 100 * An / Rn
            Rns.append(Rn)
            Ans.append(An)
            Accs.append(Acc)

        plt.plot(np.arange(1, 1 + len(Accs)), Accs, label='Accuracy')
        plt.plot(np.arange(2, 2 + len(Accs)), Ans, label='N')
        plt.plot(np.arange(1, 1 + len(Accs)), Rns, label='R')

        x_text = 0

        plt.title(label)
        plt.xlabel("No.") #
        # plt.ylabel("") #
        plt.legend()

    plt.tight_layout(w_pad=1.2, pad=3)
    # plt.suptitle(u'直接欧式距离检索图片（像素平均值）', **Kaiti)
    plt.savefig(u'Retrieval/直接欧式距离检索图片（像素平均值）.png', dpi=300)
    plt.savefig(u'Retrieval/直接欧式距离检索图片（像素平均值）.pdf', dpi=300)

    plt.show()


def main():
    # valid(80%) as gallery, and train(20%) as retrieval
    ext = ImagesExtracter("images/valid")
    rti = RetrievalImages("images/train")

    draw1(ext, rti)

    # draw2(ext, rti)


if __name__ == '__main__':
    main()
