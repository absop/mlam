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
Lucida = {'family': 'Lucida Console'}
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

    def sort_with_similarity(self, image):
        similarities, indexes = F.cosine_similarity(
            self.images, image.view(1, -1)).sort(dim=0, descending=True)

        similarities = np.around(similarities.numpy(), 3)

        paths, targets = [], []
        for i in range(len(similarities)):
            path, target = self.samples[indexes[i]]
            paths.append(path)
            targets.append(target)

        return similarities, paths, np.array(targets)


def main():
    # valid(80%) as gallery, and train(20%) as retrieval
    ext = ImagesExtracter("images/valid")
    rti = RetrievalImages("images/train")
    # Visual output
    colors = ['C1', 'C2']
    x = np.arange(1, 1 + len(ext.images))
    plt.figure(figsize=(8, 4.5))
    for i, label in enumerate(rti.classes):
        image = rti[label][:]
        S, _, T = ext.sort_with_similarity(image)

        # Compute differences of similarities
        dS = np.diff(S)
        _, R = torch.min(torch.Tensor(dS)[(S > 0.5)[:-1]], 0)

        R = R.item() + 1
        M = sum(T[:R] == rti.target)
        A = 100 * M / R

        plt.subplot(121 + i)

        for idx, c in enumerate(rti.classes):
            sieve = T == idx
            plt.plot(x[sieve], S[sieve], '.', ms=4, color=colors[idx], label=c)
        plt.plot(np.arange(2, 2 + len(dS)), dS, label='dS', linewidth=1.0)

        ylim = plt.gca().get_ylim()
        plt.vlines(R, *ylim, linewidth=0.5, color="purple", label=u'分界线')

        plt.text(R + 5, 0.30, '%d' % R, Lucida)
        plt.text(R + 5, 0.25, "%d" % M, Lucida)
        # plt.text(R + 5, 0.20, "Acc: %.2f%%" % A, Lucida)
        plt.title(label)
        plt.xlabel("R") # Rank
        plt.ylabel("S") # Similarity
        plt.legend(loc='center left', prop=Kaiti)

    # plt.suptitle(u'直接余弦相似度检索图片', **Kaiti)
    plt.tight_layout(w_pad=0.2, h_pad=0.2, pad=2)

    plt.savefig("Retrieval/直接余弦相似度检索图片.png", dpi=300)
    plt.savefig("Retrieval/直接余弦相似度检索图片.pdf", dpi=300)


if __name__ == '__main__':
    main()
