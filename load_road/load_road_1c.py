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
    Road dataset for lidar
    """
    NUM_CLASSES = 2

    def __init__(self,
                 base_dir,
                 split='train',
                 ):
        """
        :param base_dir: path to road dataset directory
        :param split: train/val
        """
        super().__init__()
        self._base_dir = base_dir
        self._lidar_dir = os.path.join(self._base_dir, 'hght')
        self._image_dir = os.path.join(self._base_dir, 'rgb')
        self._cat_dir = os.path.join(self._base_dir, 'rev_annotations')

        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []
        self.hght = []

        print(os.path.join(os.path.join(_splits_dir, split + '.txt')))
        with open(os.path.join(os.path.join(_splits_dir, split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _img = os.path.join(self._image_dir, line + ".png")
            _hght = os.path.join(self._lidar_dir, line + ".png")
            _cat = os.path.join(self._cat_dir, line + ".png")
            assert os.path.isfile(_hght)
            assert os.path.isfile(_cat)
            self.im_ids.append(line)
            self.images.append(_img)
            self.hght.append(_hght)
            self.categories.append(_cat)
         
        assert (len(self.hght) == len(self.categories))
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _hght, _target = self._make_img_gt_point_pair(index)
        composed_transforms = transforms.Compose([ transforms.ToTensor()])
        _t_hght = composed_transforms(_hght)
        #composed_transforms = transforms.Compose([ transforms.Normalize(mean=(0.4285), std=(0.197))])
        #_tn_hght = composed_transforms(_t_hght)
        _target = np.array(_target).astype(np.float32)
        _t_target = from_numpy(_target).long().view(512,512)
        sample = {'image': _t_hght, 'label': _t_target}

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
        _hght = Image.open(self.hght[index])
        _hght_padded = self._padding(_hght, 512)
        _target = Image.open(self.categories[index])
        _target_padded = self._padding(_target, 512)
        # print(_target_padded.size)
        return _hght_padded, _target_padded      


def make_data_splits_1c(base_dir, batch_size=4):
    train_set = RoadSegmentation(base_dir, split='train')
    val_set = RoadSegmentation(base_dir, split='valid')
    test_set = RoadSegmentation(base_dir, split='test')
    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader, num_class
