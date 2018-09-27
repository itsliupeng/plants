import argparse
import os
import time

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from utils import ImageDataSetWithRaw, data_transforms


use_gpu = torch.cuda.is_available()


def train(model, train_data_loader, val_data_loader, optimizer, scheduler, num_epochs, writer=None):
    for epoch_i in range(num_epochs):
        start_time = time.time()
        model.train()
        scheduler.step()

        print('Epoch {}/{}: lr {}'.format(epoch_i + 1, num_epochs, scheduler.get_lr()), end='')
        if writer:
            writer.add_scalar('lr', scheduler.get_lr()[0], global_step=epoch_i)

        running_loss = 0.0
        running_corrects = 0.0

        for inputs, labels, raw_images in train_data_loader:
            if writer:
                writer.add_image('raw-crop-label', cat_images_show(image_show(raw_images), image_show(inputs), label_show(labels)))

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, size_average=False)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(F.softmax(outputs, dim=1), 1)
            running_corrects += torch.sum(preds == labels).item()

        train_dataset_size = len(train_data_loader.dataset)
        epoch_loss = running_loss / train_dataset_size
        epoch_acc = running_corrects / train_dataset_size
        print('\t{:5s} loss {:.4f} acc {:.4f}'.format('train', epoch_loss, epoch_acc))
        train_end_time = time.time()

        val(model, val_data_loader)
        val_end_time = time.time()

        train_time = train_end_time - start_time
        val_time = val_end_time - train_end_time
        print('\ttime train {:.4f} val {:.4f}'.format(train_time, val_time))

        if writer:
            writer.add_scalar('epoch_loss', epoch_loss, global_step=epoch_i)
            writer.add_scalar('epoch_acc', epoch_acc, global_step=epoch_i)
            writer.add_scalar('epoch_train_time', train_time, global_step=epoch_i)
            writer.add_scalar('epoch_val_time', val_time, global_step=epoch_i)


def val(model, val_data_loader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0

    val_dataset_size = len(val_data_loader.dataset)

    for inputs, labels in val_data_loader:
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels, size_average=False)
        running_loss += loss.item()
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        running_corrects += torch.sum(preds == labels).item()

    epoch_loss = running_loss / val_dataset_size
    epoch_acc = running_corrects / val_dataset_size
    print('\t{:5s} loss {:.4f} acc {:.4f}'.format('val', epoch_loss, epoch_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', help='', type=str, default='/Users/liupeng/data/plants')
    parser.add_argument('-bs', '--batch_size', help='', type=int, default=4)
    parser.add_argument('-n', '--num_epoch', help='', type=int, default=30)
    args = vars(parser.parse_args())

    data_dir = args['data_dir']
    batch_size = args['batch_size']
    num_epoch = args['num_epoch']

    image_datasets = {x: ImageDataSetWithRaw(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_to_idx = image_datasets['train'].class_to_idx
    print(class_to_idx)

    train_data_loader = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_data_loader = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=12)
    model = torch.nn.DataParallel(model)
    if use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.module.parameters(), lr=1e-3)
    scheduler =  optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,80], gamma=0.1)

    tb_writer = SummaryWriter(log_dir='logs')
    train(model, train_data_loader, val_data_loader, optimizer, scheduler, num_epoch, writer=tb_writer)
    tb_writer.close()
    print('Done')

