import argparse
import os
import time
import cv2

import torch
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import ImageDataSetWithRaw, cat_image_show, data_transforms, draw_label_tensor

use_gpu = torch.cuda.is_available()
import numpy as np


# noinspection PyShadowingNames,PyShadowingNames,PyShadowingNames,PyShadowingNames,PyShadowingNames
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
                writer.add_image('raw-crop-label', cat_image_show(raw_images[0:20], inputs[0:20], draw_label_tensor(labels[0:20])))

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

        val(model, val_data_loader, epoch_i, writer)
        val_end_time = time.time()

        train_time = train_end_time - start_time
        val_time = val_end_time - train_end_time
        print('\ttime train {:.4f} val {:.4f}'.format(train_time, val_time))

        if writer:
            writer.add_scalar('loss_epoch_train', epoch_loss, global_step=epoch_i)
            writer.add_scalar('acc_epoch_train', epoch_acc, global_step=epoch_i)
            writer.add_scalar('time_epoch_train', train_time, global_step=epoch_i)
            writer.add_scalar('time_epoch_val', val_time, global_step=epoch_i)


def returnCAM(feature_conv, weight_softmax, preds):
    # generate the class activation maps upsample to 256x256
    bz, nc, height, width = feature_conv.shape
    output_cam = []
    w = weight_softmax[preds]
    f = feature_conv.reshape((bz, nc, height*width))
    assert w.size(0) == f.size(0)
    tensors = []
    for i in range(w.size(0)):
        cam = w[i].view((1, -1)).mm(f[i]).reshape((height, width))
        cam = cam - torch.min(cam)
        cam_img = cam / torch.max(cam)
        cam_img = cam_img.cpu().data.numpy()
        cam_img = np.uint8(255 * cam_img)
        heatmap = cv2.applyColorMap(cv2.resize(cam_img, (224, 224)), cv2.COLORMAP_JET)
        tensors.append(torch.Tensor(np.transpose(heatmap, (2, 0, 1))))

    result = torch.cat(tensors).reshape([len(tensors)] + list(tensors[0].shape))
    return result


# noinspection PyShadowingNames,PyShadowingNames
def val(model, val_data_loader, epoch_i=0, writer=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0

    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output)

    model.module._modules.get("layer4").register_forward_hook(hook_feature)
    # get the softmax weight
    weight_softmax = list(model.module.parameters())[-2]


    val_dataset_size = len(val_data_loader.dataset)

    for inputs, labels, _ in val_data_loader:
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels, size_average=False)
        running_loss += loss.item()
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        running_corrects += torch.sum(preds == labels).item()

        cams = returnCAM(features_blobs[0][0:8], weight_softmax, preds[0:8])
        writer.add_image('cam', torchvision.utils.make_grid(cams))

    epoch_loss = running_loss / val_dataset_size
    epoch_acc = running_corrects / val_dataset_size
    print('\t{:5s} loss {:.4f} acc {:.4f}'.format('val', epoch_loss, epoch_acc))

    if writer:
        writer.add_scalar('loss_epoch_val', epoch_loss, global_step=epoch_i)
        writer.add_scalar('acc_epoch_val', epoch_acc, global_step=epoch_i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', help='', type=str, default='/Users/liupeng/data/plants')
    parser.add_argument('-b', '--batch_size', help='', type=int, default=4)
    parser.add_argument('-n', '--num_epoch', help='', type=int, default=30)
    parser.add_argument('--num_class', help='', type=int, default=2)
    args = vars(parser.parse_args())

    data_dir = args['data_dir']
    batch_size = args['batch_size']
    num_epoch = args['num_epoch']
    num_class = args['num_class']

    image_datasets = {x: ImageDataSetWithRaw(os.path.join(data_dir, x), data_transforms[x], raw_image=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_to_idx = image_datasets['train'].class_to_idx
    print(class_to_idx)

    train_data_loader = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_data_loader = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=num_class)
    model = torch.nn.DataParallel(model)
    if use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.module.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 80], gamma=0.1)

    tb_writer = SummaryWriter(log_dir='logs')
    train(model, train_data_loader, val_data_loader, optimizer, scheduler, num_epoch, writer=tb_writer)
    tb_writer.close()
    print('Done')
