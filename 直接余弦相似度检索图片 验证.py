import os
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt

import random
import matplotlib as mpl

mpl.style.use('seaborn')

plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

data_transform = transforms.Compose([
    transforms.Resize(192),
    transforms.CenterCrop(192),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def load_image(path):
    return data_transform(Image.open(path)).view(1, -1)


def main():
    # valid(80%) as gallery, and train(20%) as retrieval
    PCs = os.listdir('images/valid/PC')
    PTs = os.listdir('images/valid/PT')
    PC = random.sample(PCs, 2)
    PT = random.sample(PTs, 2)
    for i in range(2):
        PC[i] = os.path.join('images/valid/PC', PC[i])
        PT[i] = os.path.join('images/valid/PT', PT[i])

    plt.figure(figsize=(3.6, 4.5))
    for i in range(1, 3):
        plt.subplot(2, 2, i)
        plt.imshow(Image.open(PC[i-1]))
        plt.title('PC%d'%i, fontsize=20)
        plt.axis('off')

    for i in range(1, 3):
        plt.subplot(2, 2, 2 + i)
        plt.imshow(Image.open(PT[i-1]))
        plt.title('PT%d'%i, fontsize=20)
        plt.axis('off')

    PC1 = load_image(PC[0])
    PC2 = load_image(PC[1])
    PT1 = load_image(PT[0])
    PT2 = load_image(PT[1])
    print(F.cosine_similarity(PC1, PC1, dim=1))
    print(F.cosine_similarity(PC1, PC2, dim=1))
    print(F.cosine_similarity(PC1, PT1, dim=1))
    print(F.cosine_similarity(PC1, PT2, dim=1))
    print(F.cosine_similarity(PT1, PT2, dim=1))
    print(F.cosine_similarity(PT1, PC2, dim=1))
    print(F.cosine_similarity(PT2, PC2, dim=1))

    plt.tight_layout(w_pad=0.3, h_pad=0.8, pad=0.0)

    plt.savefig('Retrieval/直接余弦相似度检索图片 验证.png', bbox_inches='tight', dpi=300)
    plt.savefig('Retrieval/直接余弦相似度检索图片 验证.pdf', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()
