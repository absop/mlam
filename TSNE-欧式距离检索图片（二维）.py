import numpy as np
import matplotlib.pyplot as plt

from utils.tsne import load_tsne2d

import matplotlib as mpl

mpl.style.use('seaborn')

plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

Kaiti = {'family': 'Kaiti'}


class RetrievalTSNE(object):
    def __init__(self, root, label=None):
        paths, tsne2d, targets, classes = load_tsne2d(root)
        tsnes = [[] for i in classes]
        for tsne, target in zip(tsne2d, targets):
            tsnes[target].append(tsne)

        self.paths = paths
        self.classes = classes
        self.targets = targets
        self.class_to_idx = {c:i for i, c in enumerate(classes)}
        self.cache = [np.array(t) for t in tsnes]
        self.label = None
        self.root = root

        if label is not None:
            self.set_label(label)

    def set_label(self, label):
        self.label = label
        self.target = self.class_to_idx[label]
        self.tsnes = self.cache[self.target]

    def __getitem__(self, index):
        if (isinstance(index, slice)):
            return self.tsnes[index].mean(0)

        if (isinstance(index, int)):
            return self.tsnes[index]

        if (isinstance(index, str)):
            self.set_label(index)
            return self

    def __len__(self):
        if self.label is not None:
            return len(self.tsnes)

        return None


def main():
    paths, tsne2d, targets, classes = load_tsne2d('images/valid')
    rt = RetrievalTSNE('images/train')
    x = np.arange(1, 1 + len(tsne2d))
    colors = ['C1', 'C2']
    for i, label in enumerate(rt.classes):
        items = []
        qs = rt[label]
        q = qs[:5]
        plt.subplot(221 + i)
        for t, target in zip(tsne2d, targets):
            items.append((((t - q)**2).sum(), target))
        items.sort(key=lambda x: x[0])
        D = np.array([it[0] for it in items])
        T = np.array([it[1] for it in items])
        dD = np.diff(D)
        R = max(enumerate(dD), key=lambda x: x[1])[0] + 1
        A = 100.0 * sum(T[:R] == i) / R

        # plt.plot(np.arange(len(D)), D, label='dist')
        for j, c in enumerate(rt.classes):
            sieve = T == j
            plt.plot(x[sieve], D[sieve], '.', ms=4, color=colors[j], label=c)
        plt.plot(np.arange(len(dD)), dD, label='d(dist)')

        ylim = plt.gca().get_ylim()
        plt.vlines(R, *ylim, color="purple", linewidth=0.5, label='分界线')
        plt.text(R + 5, sum(ylim)/2 * 0.9, '%d' % R)
        plt.text(R + 5, sum(ylim)/2 * 0.7, '%.2f$\\%%$' % A)

        plt.legend(loc='upper left', prop=Kaiti)
        plt.xlabel('rank')
        plt.ylabel('dist')
        plt.title(['(a)', '(b)'][i] + '. %s' % label)

        # 轮流输入每张图像
        plt.subplot(223 + i)
        Rns = []
        Ans = []
        Accs = []
        for j in range(len(qs)):
            q = qs[j]
            items = []
            for t, target in zip(tsne2d, targets):
                items.append((((t - q)**2).sum(), target))
            items.sort(key=lambda x: x[0])
            D = np.array([it[0] for it in items])
            dD = np.diff(D)
            R = max(enumerate(dD), key=lambda x: x[1])[0] + 1
            An = [it[1] for it in items[:R]].count(rt.target)
            Rns.append(R)
            Ans.append(An)
            Accs.append(100 * An / R)

        # plt.plot(np.arange(len(Ans)), Ans,
        #     label='Accuracy number')
        # plt.plot(np.arange(len(Rns)), Rns,
        #     label='Retrieve number')
        plt.gca().set_ylim(0, 150)
        plt.plot(np.arange(len(Accs)), Accs, color=colors[i], label='Acc.')

        plt.legend(loc='upper left')
        plt.xlabel('No.')
        plt.ylabel('%%')
        plt.title(['(c)',  '(d)'][i] + '. %s' % label)

    plt.tight_layout(w_pad=1.2, pad=1.2)
    # plt.suptitle(u'二维TSNE-欧式距离检索图片', **Kaiti)
    plt.savefig('TSNE/TSNE-欧式距离检索图片（二维）.png', dpi=300)
    plt.savefig('TSNE/TSNE-欧式距离检索图片（二维）.pdf', dpi=300)


if __name__ == '__main__':
    main()
