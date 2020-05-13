import os, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
from PIL  import Image

plt.rc('font', family='serif')


def train(model, loss_fn, optimizer, train_loader):
    model.train()

    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        predict = model(inputs.requires_grad_())

        loss = loss_fn(predict, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def load_model(checkpoints):
    model = models.vgg16(pretrained=True).cuda()

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2).cuda()

    filepath = os.path.join(checkpoints, 'checkpoint.pth.tar')
    try:
        state_dict = torch.load(filepath)
        model.load_state_dict(state_dict['model'])
    except:
        train(model,
            nn.CrossEntropyLoss(),
            optim.Adam(model.parameters(), lr=0.001),
            train_loader)
        torch.save({'model': model.state_dict()}, filepath)

    return model


class myDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample, target


data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def make_dataset_loader(root, batch_size, shuffle, Dataset=None):
    Dataset = Dataset or datasets.ImageFolder
    return torch.utils.data.DataLoader(
        Dataset(root=root, transform=data_transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1)


"""
Dataset layout:
/images
    /train  20%(101 images)
    /valid  80%(406 images)

"""

train_loader = make_dataset_loader('images/train', 8, True)
valid_loader = make_dataset_loader('images/valid', 32, True, myDataset)


# Set pretrained to True to perform transfer learning.
def main():
    model = load_model('classifier')

    model.eval()
    with torch.no_grad():
        for paths, inputs, targets in valid_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            predict = model(inputs)
            targets = torch.max(predict, 1)[1]
            classes = [valid_loader.dataset.classes[i] for i in targets]

            # for p, c in zip(paths, classes):
            #     print('%s: %s' % (p, c))

            for i, path in enumerate(paths):
                plt.subplot(4, 8, i + 1)
                plt.imshow(Image.open(path))
                plt.title(classes[i], y=-0.32)
                plt.xticks([])
                plt.yticks([])

            plt.tight_layout(w_pad=-0.1, h_pad=-4, pad=0.5)
            plt.savefig('classifier 示例.png', dpi=300)
            plt.savefig('classifier 示例.pdf', dpi=300)

            plt.show()

            break



if __name__ == '__main__':
    main()
