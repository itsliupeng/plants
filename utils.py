import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import os


class ImageDataSetWithRaw(ImageFolder):
    def __init__(self, root, transform, raw_image=False):
        super(ImageDataSetWithRaw, self).__init__(root, transform)
        self.to_tensor = self.to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.raw_image = raw_image

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample_aug = self.transform(sample)

        if self.raw_image:
            return sample_aug, target, self.to_tensor(sample)
        else:
            return sample_aug, target


def draw_label_image(text: str, size=(224, 224)):
    # make a blank image for the text, initialized to transparent text color
    img = Image.new('RGB', size, (0, 0, 0))
    d = ImageDraw.Draw(img)

    # to-do: how to align
    if len(text) <= 2:
        font_size = 100
        xy = (80, 60)
    else:
        font_size = 40
        xy = (60, 90)

    # get a font
    fnt = ImageFont.truetype('data/fonts/FreeMono.ttf', size=font_size)
    # get a drawing context
    d.text(xy, text, font=fnt, fill=(255, 255, 255))
    return img


to_tensor = torchvision.transforms.ToTensor()


def draw_label_tensor(label: torch.Tensor, size=(224, 224)):
    return torch.cat([to_tensor(draw_label_image(str(i), size)) for i in label.data.cpu().numpy()]).reshape((-1, 3, size[0], size[1]))


def cat_image_show(*tensors):
    images = [make_grid(i.cpu(), nrow=10, normalize=True) for i in tensors]
    total = torch.cat(images).reshape([len(images)] + list(images[0].shape))
    return torchvision.utils.make_grid(total, nrow=1, normalize=True, scale_each=True)


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
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


def save_ckpt(output_dir, model, optimizer, epoch, batch_size):
    """Save checkpoint"""
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_epoch{}.pth'.format(epoch))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save({
        'epoch': epoch,
        'batch_size': batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_name)

    print(f'save model {save_name} in {output_dir}')