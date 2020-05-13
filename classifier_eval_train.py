import shutil, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

from utils import AverageMeter, Accuracy
from utils import Logger, subplot, savefig


def train(model, loss_fn, optimizer, train_loader):
    model.train()

    losses = AverageMeter()
    accuracy = Accuracy()
    start = time.time()

    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        predict = model(inputs.requires_grad_())

        # print("train: input.requires_grad is ", inputs.requires_grad)

        loss = loss_fn(predict, targets)
        losses.update(loss.item(), inputs.size(0))
        accuracy.update(torch.max(predict, 1)[1], targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end = time.time()

    return losses, accuracy, end - start


def valid(model, loss_fn, valid_loader):
    model.eval()

    losses = AverageMeter()
    accuracy = Accuracy()
    start = time.time()

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            predict = model(inputs)

            # print("valid: input.requires_grad is ", inputs.requires_grad)

            loss = loss_fn(predict, targets)
            losses.update(loss.item(), inputs.size(0))
            accuracy.update(torch.max(predict, 1)[1], targets)

    end = time.time()

    return losses, accuracy, end - start


def save_checkpoint(state, is_best, checkpoints):
    filepath = os.path.join(checkpoints, 'checkpoint.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(
            filepath,
            os.path.join(checkpoints, 'model_best.pth.tar'))


def load_checkpoint(model, optimizer, checkpoints):
    start_epoch = 1
    best_accuracy = Accuracy()

    filepath = os.path.join(checkpoints, 'checkpoint.pth.tar')
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        best_accuracy = Accuracy(checkpoint['best_accuracy'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return start_epoch, best_accuracy


def train_loop(model, loss_fn, optimizer, train_loader, valid_loader, **kwargs):
    checkpoints = kwargs['checkpoints']
    start_epoch, best_accuracy = load_checkpoint(model, optimizer, checkpoints)

    logger = kwargs['logger']
    resume =  os.path.isfile(logger[0])
    logger = Logger(*logger, resume=resume)
    if not resume:
        logger.set_names(
            ['Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    average_train_time = AverageMeter()
    average_valid_time = AverageMeter()

    n_epochs = kwargs['n_epochs']

    for epoch in range(start_epoch, n_epochs + 1):
        print("=================epoch: {}=================".format(epoch))

        train_loss, train_accuracy, train_time = train(
            model, loss_fn, optimizer, train_loader)
        train_loss, train_accuracy, train_time = valid(model, loss_fn, train_loader)
        valid_loss, valid_accuracy, valid_time = valid(model, loss_fn, valid_loader)
        logger.append([train_loss.avg, valid_loss.avg, train_accuracy.value, valid_accuracy.value])

        is_best = valid_accuracy > best_accuracy
        best_accuracy = max(valid_accuracy, best_accuracy)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': str(best_accuracy),
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoints)

        average_train_time.update(train_time, 1)
        average_valid_time.update(valid_time, 1)
        print("Train loss: {}, accuracy: {}, time: {:.2f}s".format(train_loss, train_accuracy, train_time))
        print("Test loss: {}, accuracy: {}|(best: {}), time: {:.2f}s".format(valid_loss, valid_accuracy, best_accuracy, valid_time))

        print()

    logger.plot()
    logger.savefig('log.eps')
    logger.savefig('log.pdf')
    logger.savefig('log.png', dpi=300)
    logger.close()

    print("Train time: {}s, Test time: {}s".format(
        average_train_time, average_valid_time))



data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def make_dataset_loader(path, batch_size, shuffle):
    dataset = datasets.ImageFolder(root=path, transform=data_transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)


"""
Dataset layout:
Classifier/Data
    /train  20%(101 images)
    /valid  80%(406 images)

"""

train_loader = make_dataset_loader('Classifier/Data/train', 8, True)
valid_loader = make_dataset_loader('Classifier/Data/valid', 8, False)


training_task_map = [
    {
        "n_epochs": 10,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "initial": {"pretrained": True},
        "checkpoints": "Classifier/checkpoints/eval-train/pretrained",
        "logger": ('Classifier/checkpoints/eval-train/pretrained/log.txt', "Pretrained VGG16 network", '训练次数')
    },
    {
        "n_epochs": 20,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "initial": {"pretrained": False, "num_classes": 2},
        "checkpoints": "Classifier/checkpoints/eval-train",
        "logger": ('Classifier/checkpoints/eval-train/log.txt', "VGG16 network", '训练次数')
    }
]

# Set pretrained to True to perform transfer learning.
def main():
    for task in training_task_map:
        model = models.vgg16(**task["initial"]).cuda()

        os.makedirs(task['checkpoints'], exist_ok=True)

        # params have requires_grad=True by default
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 2).cuda()

        train_loop(
            model,
            nn.CrossEntropyLoss(),
            optim.Adam(model.parameters(), lr=0.001),
            **task)


if __name__ == '__main__':
    main()

    subplot([net['logger'] for net in training_task_map])
    savefig("Classifier/VGG16 training process - eval-train.png")
    savefig("Classifier/VGG16 training process - eval-train.pdf")
