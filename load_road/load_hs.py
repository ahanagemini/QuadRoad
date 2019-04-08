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
from skimage.external import tifffile
from torchvision_x.transforms import functional as F

class RoadSegmentation(Dataset):
    """
    Road dataset
    """
    NUM_CLASSES = 2

    def __init__(self,
                 base_dir,
                 split='train',
                 ):
        """
        :param base_dir: path to road dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._hs123_dir = os.path.join(self._base_dir, 'data_hsi_split/123')
        self._hs456_dir = os.path.join(self._base_dir, 'data_hsi_split/456')
        self._hs78_dir = os.path.join(self._base_dir, 'data_hsi_split/78')
        self._cat_dir = os.path.join(self._base_dir, 'rev_annotations')

        _splits_dir = self._base_dir

        self.im_ids = []
        self.hs123 = []
        self.categories = []
        self.hs456 = []
        self.hs78 = []

        print(os.path.join(os.path.join(_splits_dir, split + '.txt')))
        with open(os.path.join(os.path.join(_splits_dir, split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _hs123 = os.path.join(self._hs123_dir, line + ".tif")
            _hs456 = os.path.join(self._hs456_dir, line + ".tif")
            _hs78 = os.path.join(self._hs78_dir, line + ".tif")
            _cat = os.path.join(self._cat_dir, line + ".png")
            assert os.path.isfile(_hs123)
            assert os.path.isfile(_hs456)
            assert os.path.isfile(_hs78)
            assert os.path.isfile(_cat)
            self.im_ids.append(line)
            self.hs123.append(_hs123)
            self.hs456.append(_hs456)
            self.hs78.append(_hs78)
            self.categories.append(_cat)
         
        assert (len(self.hs123) == len(self.categories))
        assert (len(self.hs456) == len(self.categories))
        assert (len(self.hs78) == len(self.categories))
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.hs123)))

    def __len__(self):
        return len(self.hs123)


    def __getitem__(self, index):
        _hs123, _hs456, _hs78, _target = self._make_img_gt_point_pair(index)
        _hs123 = _hs123.astype(dtype=np.int32)
        _hs456 = _hs456.astype(dtype=np.int32)
        _hs78 = _hs78.astype(dtype=np.int32)
        _hs78 = _hs78[:,:,:2]
        #composed_transforms = transforms_seg.SegCompose([ transforms_seg.SegToTensor()])
        _t_hs123 = F.to_tensor(_hs123)
        _t_hs456 = F.to_tensor(_hs456)
        _t_hs78 = F.to_tensor(_hs78)
        #_t_hght = composed_transforms(_hght)
        #print(_t_hs123.shape)
        #print(_t_hs456.shape)
        #print(_t_hs78.shape)
        _t_img = cat((_t_hs123,_t_hs456,_t_hs78),0)
        #composed_transforms = transforms.Compose([transforms.Normalize(mean=(0.339, 0.336, 0.302), std=(0.056, 0.041, 0.021))])
        #_tn_img = composed_transforms(_t_img)
        _target = np.array(_target).astype(np.float32)
        _t_target = from_numpy(_target).long().view(512,512)
        #print(_t_img.shape)
        #print(_t_target.shape)
        sample = {'image': _t_img, 'label': _t_target}

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
        #_img = Image.open(self.images[index]).convert('RGB')
        #_hght = Image.open(self.hght[index])
        #_img_padded = self._padding(_img, 512)
        #_hght_padded = self._padding(_hght, 512)
        _hs123 = tifffile.imread(self.hs123[index])
        _hs456 = tifffile.imread(self.hs456[index])
        _hs78 = tifffile.imread(self.hs78[index])
        _target = Image.open(self.categories[index])
        _target_padded = self._padding(_target, 512)
        # print(_target_padded.size)
        return _hs123, _hs456, _hs78, _target_padded      


def make_data_splits_hs(base_dir, batch_size=4):
    train_set = RoadSegmentation(base_dir, split='train')
    val_set = RoadSegmentation(base_dir, split='valid')
    test_set = RoadSegmentation(base_dir, split='test')
    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader, num_class
