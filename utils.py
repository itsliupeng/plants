import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid


class ImageDataSetWithRaw(ImageFolder):
    def __init__(self, root, transform):
        super(ImageDataSetWithRaw, self).__init__(root, transform)
        self.to_tensor = self.to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample_raw = self.to_tensor(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, sample_raw


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
    return torch.cat([to_tensor(draw_label_image(str(i), size)) for i in label.numpy()]).reshape((-1, 3, size[0], size[1]))


def cat_image_show(*tensors):
    total = torch.cat([make_grid(i, nrow=10, normalize=True) for i in tensors])
    new_shape = [len(tensors)] + list(tensors[0].shape)
    total = total.reshape(new_shape)
    return torchvision.utils.make_grid(total, nrow=1, normalize=True, scale_each=True)


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
