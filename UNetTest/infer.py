import torch
from torchvision.io import read_image
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

from unet import UNet


def preprocess(img):
    img = img / 127.5 - 1
    img = TF.resize(img, size=[512, 512], interpolation=TF.InterpolationMode.BILINEAR)
    return img


@torch.no_grad()
def infer(model_path, img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=3, n_classes=1, bilinear=True).eval()
    checkpoint_data = torch.load(model_path)
    model.load_state_dict(checkpoint_data['unet_state_dict'])

    img_width = img.size(-1)
    img_height = img.size(-2)

    src = preprocess(img)

    src = src.to(device)
    model.to(device)

    pred = model(src)

    pred_hard = ((torch.sign(pred) + 1) / 2).cpu()

    pred_hard = TF.resize(pred_hard, size=[img_height, img_width], interpolation=TF.InterpolationMode.NEAREST_EXACT)

    return pred_hard


def main():
    img_path = 'D:/Dataset/ISIC_2017_Task_1/ISIC-2017_Test_v2_Data/ISIC_0015046.jpg'
    img = read_image(img_path)
    img = img.unsqueeze(0)

    pred = infer(model_path='./checkpoints/unet_checkpoint_450.pth', img=img)

    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze(0).permute([1, 2, 0]))
    plt.subplot(1, 2, 2)
    plt.imshow(pred.squeeze(0, 1))
    plt.show()


if __name__ == '__main__':
    main()
