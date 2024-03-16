from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as TF

import pandas as pd


class ISICDataset(Dataset):
    def __init__(self, dataset_root, usage='train', resize_size=(512, 512)):
        self.dataset_root = dataset_root
        self.usage = usage
        self.resize_size = resize_size

        if self.usage == 'train':
            self.src_root = dataset_root + '/' + 'ISIC-2017_Training_Data'
            self.tgt_root = dataset_root + '/' + 'ISIC-2017_Training_Part1_GroundTruth'
            self.meta_path = self.src_root + '/' + 'ISIC-2017_Training_Data_metadata.csv'

        elif self.usage == 'val':
            self.src_root = dataset_root + '/' + 'ISIC-2017_Validation_Data'
            self.tgt_root = dataset_root + '/' + 'ISIC-2017_Validation_Part1_GroundTruth'
            self.meta_path = self.src_root + '/' + 'ISIC-2017_Validation_Data_metadata.csv'

        elif self.usage == 'test':
            self.src_root = dataset_root + '/' + 'ISIC-2017_Test_v2_Data'
            self.tgt_root = dataset_root + '/' + 'ISIC-2017_Test_v2_Part1_GroundTruth'
            self.meta_path = self.src_root + '/' + 'ISIC-2017_Test_v2_Data_metadata.csv'

        meta_csv = pd.read_csv(self.meta_path)
        self.image_id_list = list(meta_csv['image_id'])

    def __getitem__(self, index):
        src_img_path = self.src_root + '/' + self.image_id_list[index] + '.jpg'
        tgt_img_path = self.tgt_root + '/' + self.image_id_list[index] + '_segmentation.png'

        src_img = read_image(src_img_path)
        tgt_img = read_image(tgt_img_path)

        src_img = TF.resize(src_img, self.resize_size, interpolation=TF.InterpolationMode.BILINEAR)
        tgt_img = TF.resize(tgt_img, self.resize_size, interpolation=TF.InterpolationMode.NEAREST_EXACT)

        src_img = src_img / 127.5 - 1.0
        tgt_img = tgt_img / 255

        return src_img, tgt_img

    def __len__(self):
        return len(self.image_id_list)
