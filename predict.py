import argparse
import os
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import ImageDataSetWithName, data_transforms, cam_tensor, draw_label_tensor,cat_image_show
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from functools import reduce

use_gpu = torch.cuda.is_available()


def predict(model, data_loader, writer=None):
    model.eval()

    # hook the feature extractor
    def hook_feature(module, input, output):
        global features_blobs
        features_blobs = output


    model.module._modules.get("layer4").register_forward_hook(hook_feature)
    # get the softmax weight
    weight_softmax = list(model.module.parameters())[-2]

    pred_result = []
    with torch.no_grad():
        for idx, (inputs, names) in enumerate(data_loader):
            if use_gpu:
                inputs = inputs.cuda()

            outputs = model(inputs)
            softmax = F.softmax(outputs, dim=1)
            one_prob = softmax[:, 1]
            _, preds = torch.max(softmax, 1)

            if writer and idx % write_image_freq == 0:
                cams = cam_tensor(inputs[0:20].cpu().numpy(), features_blobs[0:20].cpu().numpy(), weight_softmax[preds[0:20]].data.cpu().numpy())
                total_image = cat_image_show(inputs[0:20], cams, draw_label_tensor(preds[0:20]), draw_label_tensor(one_prob[0:20]))
                writer.add_image('image_raw_pred', total_image, global_step=idx)

            pred_result.append(list(zip(names, ['{:.3f}'.format(i) for i in one_prob.cpu().numpy()])))

    pred_result = reduce(lambda a, b: a + b, pred_result)

    return pred_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', help='', type=str, default='/Users/liupeng/data/plants/test')
    parser.add_argument('-b', '--batch_size', help='', type=int, default=4)
    parser.add_argument('--output_dir', help='', type=str, default=os.getcwd())
    parser.add_argument('--model_path', help='', type=str, default='')
    parser.add_argument('--write_image_freq', help='', type=int, default=10)

    args = vars(parser.parse_args())
    print(f'args: {args}')

    data_dir = args['data_dir']
    batch_size = args['batch_size']
    output_dir = args['output_dir']
    model_path = args['model_path']
    write_image_freq = args['write_image_freq']

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)

    if os.path.join(model_path):
        state_dict = torch.load(model_path)['model']
        model.load_state_dict(state_dict)
        print(f'loaded model weights from {model_path}')

    model = torch.nn.DataParallel(model)
    if use_gpu:
        model = model.cuda()

    data_loader = DataLoader(ImageDataSetWithName(data_dir, data_transforms['val']), batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    tb_writer = SummaryWriter(log_dir='logs')
    pred_result = predict(model, data_loader, tb_writer)
    tb_writer.close()

    with open(os.path.join(output_dir, 'predict_result.csv'), 'w') as f:
        f.write('id,prob\n')
        for (id, prob) in pred_result:
            f.write(f'{id},{prob}\n')

    print('Done')

