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
            _image = os.path.join(self._image_dir, line + ".png")
            _hght = os.path.join(self._lidar_dir, line + ".png")
            _cat = os.path.join(self._cat_dir, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_hght)
            assert os.path.isfile(_cat)
            self.im_ids.append(line)
            self.images.append(_image)
            self.hght.append(_hght)
            self.categories.append(_cat)
         
        assert (len(self.images) == len(self.categories))
        assert (len(self.hght) == len(self.categories))
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def transform_rgb(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            transforms.ToTensor()])
            #transforms.Normalize(mean=(0.339, 0.336, 0.302), std=(0.056, 0.041, 0.021))])
        print("Transforming")
        return composed_transforms(sample)
    
    def transform_hght(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            transforms.ToTensor()])
            #transforms.Normalize((0.4285),(0.197))])
        print("Transforming")
        return composed_transforms(sample)

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _hght, _target = self._make_img_gt_point_pair(index)
        composed_transforms = transforms.Compose([ transforms.ToTensor()])
        _t_img = composed_transforms(_img)
        _t_hght = composed_transforms(_hght)
        #print(_t_img.shape)
        #print(_t_hght.shape)
        _t_imhg = cat((_t_img,_t_hght),0)
        composed_transforms = transforms.Compose([ transforms.Normalize(mean=(0.339, 0.336, 0.302, 0.4285), std=(0.056, 0.041, 0.021, 0.197))])
        _tn_imhg = composed_transforms(_t_imhg)
        _target = np.array(_target).astype(np.float32)
        _t_target = from_numpy(_target).long().view(512,512)
        #print(_tn_imhg.shape)
        # print(_t_target.shape)
        sample = {'image': _tn_imhg, 'label': _t_target}

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
        _hght = Image.open(self.hght[index])
        _img_padded = self._padding(_img, 512)
        _hght_padded = self._padding(_hght, 512)
        _target = Image.open(self.categories[index])
        _target_padded = self._padding(_target, 512)
        # print(_target_padded.size)
        return _img_padded, _hght_padded, _target_padded      


def make_data_splits_4c(base_dir, batch_size=4):
    train_set = RoadSegmentation(base_dir, split='train')
    val_set = RoadSegmentation(base_dir, split='valid')
    test_set = RoadSegmentation(base_dir, split='test')
    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader, num_class
