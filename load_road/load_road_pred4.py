from __future__ import print_function, division
import os
from PIL import Image
from PIL import ImageOps
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import from_numpy
from torch import cat

class RoadSegmentation(Dataset):
    """
    Road dataset for 4 predictions from rgb, hght, hs and shallow 17class to 2 class models
    Load 4 channels from 4 different sources
    """
    def __init__(self,
                 base_dir,
                 split='train',
                 ):
        """
        :param base_dir: path to road dataset directory
        :param split: train/val
        :num_classes: number of target classes
        :cat_dir: directory that stores the labels
        :norm: whether we use normalization or not. Values are 0 or 1.
        :split: The data split to be used
        :purpose: if 'train' then shuffle else do not shuffle

        """
        super().__init__()
        self._base_dir = base_dir
        self._lidar_dir = os.path.join(self._base_dir, 'pred_hght_heat_c')
        self._image_dir = os.path.join(self._base_dir, 'pred_rgb_heat_c')
        self._hs_dir = os.path.join(self._base_dir, 'pred_hs_heat_c')
        self._p17_1_dir = os.path.join(self._base_dir, 'pred_17_1_heat_c')
        self._cat_dir = os.path.join(self._base_dir, 'rev_annotations')

        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []
        self.hght = []
        self.hs = []
        self.p17_1 = []

        print(os.path.join(os.path.join(_splits_dir, split + '.txt')))
        with open(os.path.join(os.path.join(_splits_dir, split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(self._image_dir, line + ".png")
            _hght = os.path.join(self._lidar_dir, line + ".png")
            _hs = os.path.join(self._hs_dir, line + ".png")
            _p17_1 = os.path.join(self._p17_1_dir, line + ".png")
            _cat = os.path.join(self._cat_dir, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_hght)
            assert os.path.isfile(_hs)
            assert os.path.isfile(_p17_1)
            assert os.path.isfile(_cat)
            self.im_ids.append(line)
            self.images.append(_image)
            self.hght.append(_hght)
            self.hs.append(_hs)
            self.p17_1.append(_p17_1)
            self.categories.append(_cat)
         
        assert (len(self.images) == len(self.categories))
        assert (len(self.hght) == len(self.categories))
        assert (len(self.hs) == len(self.categories))
        assert (len(self.p17_1) == len(self.categories))
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _hght, _hs, _p17_1, _target = self._make_img_gt_point_pair(index)
        composed_transforms = transforms.Compose([ transforms.ToTensor()])
        _t_img = composed_transforms(_img)
        _t_hght = composed_transforms(_hght)
        _t_hs = composed_transforms(_hs)
        _t_p17_1 = composed_transforms(_p17_1)
        _t_input = cat((_t_img,_t_hght, _t_hs,_t_p17_1),0)
        #composed_transforms = transforms.Compose([ transforms.Normalize(mean=(0.339, 0.336, 0.302, 0.4285), std=(0.056, 0.041, 0.021, 0.197))])
        #_tn_imhg = composed_transforms(_t_imhg)
        _target = np.array(_target).astype(np.float32)
        _t_target = from_numpy(_target).long().view(512,512)
        sample = {'image': _t_input, 'label': _t_target}

        return sample

    def _padding(self,img,expected_size):
        desired_size = expected_size
        delta_width = desired_size - img.size[0]
        delta_height = desired_size - img.size[1]
        pad_width = delta_width //2
        pad_height = delta_height //2
        padding = (pad_width,pad_height,delta_width-pad_width,delta_height-pad_height)
        return ImageOps.expand(img, padding)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index])
        _hght = Image.open(self.hght[index])
        _hs = Image.open(self.hs[index])
        _p17_1 = Image.open(self.p17_1[index])
        _img_padded = self._padding(_img, 512)
        _hght_padded = self._padding(_hght, 512)
        _hs_padded = self._padding(_img, 512)
        _p17_1_padded = self._padding(_hght, 512)
        _target = Image.open(self.categories[index])
        _target_padded = self._padding(_target, 512)
        return _img_padded, _hght_padded, _hs_padded, _p17_1_padded, _target_padded      


def make_data_splits_p4(base_dir, batch_size=4):
    train_set = RoadSegmentation(base_dir, split='train')
    val_set = RoadSegmentation(base_dir, split='valid')
    test_set = RoadSegmentation(base_dir, split='test')
    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader, num_class
