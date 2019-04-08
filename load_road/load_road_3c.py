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
    Road dataset for 3 channels
    """

    def __init__(self,
                 base_dir,
                 num_classes,
                 cat_dir,
                 norm,
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
        self._cat_dir = os.path.join(self._base_dir, cat_dir)
        self._num_classes = num_classes
        self._norm = norm
        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []
        self.hght = []

        print(os.path.join(os.path.join(_splits_dir, split + '.txt')))
        with open(os.path.join(os.path.join(_splits_dir, split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(self._image_dir, line + ".png")
            _hght = os.path.join(self._lidar_dir, line + ".png")
            _cat = os.path.join(self._cat_dir, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_hght)
            assert os.path.isfile(_cat)
            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)
         
        assert (len(self.images) == len(self.categories))
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        composed_transforms = transforms.Compose([ transforms.ToTensor()])
        _t_img = composed_transforms(_img)
        if self._norm == 1:
            composed_transforms = transforms.Compose([transforms.Normalize(mean=(0.339, 0.336, 0.302), std=(0.237, 0.201, 0.160))])
            _tn_img = composed_transforms(_t_img)
        else:
            _tn_img = _t_img
        _target = np.array(_target).astype(np.float32)
        _t_target = from_numpy(_target).long().view(512,512)
        sample = {'image': _tn_img, 'label': _t_target}

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
        _img = Image.open(self.images[index]).convert('RGB')
        _img_padded = self._padding(_img, 512)
        _target = Image.open(self.categories[index])
        _target_padded = self._padding(_target, 512)
        # print(_target_padded.size)
        return _img_padded, _target_padded      


def make_data_splits_3c(base_dir, num_class, cat_dir, norm, batch_size=4):
    train_set = RoadSegmentation(base_dir, num_class, cat_dir, norm, split='train')
    val_set = RoadSegmentation(base_dir, num_class, cat_dir, norm, split='valid')
    test_set = RoadSegmentation(base_dir, num_class, cat_dir, norm, split='test')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader, num_class
