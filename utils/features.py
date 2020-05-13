import os
import torch
from torchvision import datasets, models, transforms


tf = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class QueriesFolder(datasets.ImageFolder):
    def __init__(self, root, label=None, transform=tf):
        super(QueriesFolder, self).__init__(root=root, transform=transform)
        self.label = None

        if label is not None:
            self.set_label(label)

    def set_label(self, label):
        if self.label == label:
            return

        index = self.class_to_idx[label]

        self.samples = [s for s in self.imgs if s[1] == index]
        self.target = index
        self.label = label


class GalleryLoader(torch.utils.data.DataLoader):
    def __init__(self, root, transform=tf, batch_size=8, shuffle=False):
        super(GalleryLoader, self).__init__(
            dataset=datasets.ImageFolder(root=root, transform=transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0)


class QueriesLoader(torch.utils.data.DataLoader):
    def __init__(self, root,
                 label=None, transform=tf, batch_size=8, shuffle=True):
        super(QueriesLoader, self).__init__(
            dataset=QueriesFolder(root=root, label=label, transform=transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0)


def extract_features(model, data_loader):
    features = torch.FloatTensor()

    with torch.no_grad():
        for imgs, paths in data_loader:
            imgs = imgs.cuda()
            outputs = model(imgs)

            ff = outputs.data.cpu()
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff), 0)

    return features


def load_gallery_features(model, root, data_loader=None):
    filepath = os.path.join(root, "gallery.features.pth.tar")
    try:
        features = torch.load(filepath)
    except:
        data_loader = data_loader or GalleryLoader(root)
        features = extract_features(model, data_loader)
        torch.save(features, filepath)

    return features


def load_queries_features(model, root, label, data_loader=None):
    filepath = os.path.join(root, "queries.features.pth.tar")
    try:
        state_dict = torch.load(filepath)
    except:
        state_dict = {}

    if label in state_dict:
        features = state_dict[label]
    else:
        data_loader = data_loader or QueriesLoader(root, label)
        features = extract_features(model, data_loader)
        state_dict[label] = features

        torch.save(state_dict, filepath)

    return features


def pretrained_vgg16():
    model = models.vgg16(pretrained=True).cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


def load_gallery_features_vgg16(root):
    return load_gallery_features(pretrained_vgg16(), root)


def load_queries_features_vgg16(root, label):
    return load_queries_features(pretrained_vgg16(), root, label)
