import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import argparse
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

use_gpu = torch.cuda.is_available()


def train(model, train_data_loader, val_data_loader, optimizer, scheduler, num_epochs, dataset_sizes):
    for epoch in range(num_epochs):
        model.train()
        print('Epoch {}/{}: '.format(epoch + 1, num_epochs), end='')
        scheduler.step()
        print('lr ', scheduler.get_lr(), end='')

        running_loss = 0.0
        running_corrects = 0.0

        for inputs, labels in train_data_loader:
            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, size_average=False)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects / dataset_sizes['train']
        print('\t{:5s} loss {:.4f} acc {:.4f}'.format('train', epoch_loss, epoch_acc))
        val(model, val_data_loader)


def val(model, val_data_loader, dataset_sizes):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in val_data_loader:
        inputs = Variable(inputs)
        labels = Variable(labels)
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        running_loss += loss.data[0]
        _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['val']
    epoch_acc = running_corrects / dataset_sizes['val']
    print('\t{:5s} loss {:.4f} acc {:.4f}'.format('val', epoch_loss, epoch_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', help='', type=str)
    parser.add_argument('-bs' '--batch_size', help='', type=int)
    parser.add_argument('-n' '--num_epoch', help='', type=int)
    args = vars(parser.parse_args())

    data_dir = args['data_dir']
    batch_size = args['batch_size']
    num_epoch = args['num_epoch']

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    train_data_loader = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_data_loader = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = torchvision.models.resnet101(pretrained=True)
    model.fc = nn.Linear(in_features=2018, out_features=12)

    optimizer = optim.Adam(model.module.fc.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train(model, train_data_loader, val_data_loader, optimizer, scheduler, num_epoch, dataset_sizes)

    print('Done')



