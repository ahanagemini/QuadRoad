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
from torch import max as tmax

class RGB(Dataset):
    """
    Road dataset for 3 channels rgb only.
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
        :num_classes: number of target classes
        :cat_dir: directory that stores the labels
        :norm: whether we use normalization or not. Values are 0 or 1.
        :split: The data split to be used
        :purpose: if 'train' then shuffle else do not shuffle
        """
        super().__init__()
        self._base_dir = base_dir
        if split == 'train_aug':
            self._image_dir = os.path.join(self._base_dir, 'rgb_augment')
            self._cat_dir = os.path.join(self._base_dir, cat_dir)
        else:
            self._image_dir = os.path.join(self._base_dir, 'rgb')
            self._cat_dir = os.path.join(self._base_dir, cat_dir)
        self._num_classes = num_classes
        self._norm = norm
        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []

        print(os.path.join(os.path.join(_splits_dir, split + '.txt')))
        with open(os.path.join(os.path.join(_splits_dir, split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(self._image_dir, line + ".png")
            _cat = os.path.join(self._cat_dir, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)
         
        assert (len(self.images) == len(self.categories))
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        """
        Function to compute the number of images
        Returns:
            Length of the images list
        """
        return len(self.images)


    def __getitem__(self, index):
        """
        Function to return an image, target pair
        Args:
            index: index of th epair to be returned
        Returns:
            image, label pair
        """
        _img, _target = self._make_img_gt_point_pair(index)
        composed_transforms = transforms.Compose([ transforms.ToTensor()])
        _t_img = composed_transforms(_img)
        #print(tmax(_t_img))
        if self._norm:
            composed_transforms = transforms.Compose(
                    [transforms.Normalize(mean=(0.339, 0.336, 0.302),
                        std=(0.237, 0.201, 0.160))])
            _tn_img = composed_transforms(_t_img)
        else:
            _tn_img = _t_img
        #print(tmax(_t_img))
        #print(tmax(_tn_img))
        _target = np.array(_target).astype(np.float32)
        _t_target = from_numpy(_target).long().view(512,512)
        sample = {'image': _tn_img, 'label': _t_target}

        return sample

    def _padding(self,img,expected_size):
        """
        Function to add padding to images
        Args:
            img: The image before padding
            expected_size: Img size after padding
        Returns: Padded image
        """
        desired_size = expected_size
        delta_width = desired_size - img.size[0]
        delta_height = desired_size - img.size[1]
        pad_width = delta_width //2
        pad_height = delta_height //2
        padding = (pad_width,pad_height, delta_width - pad_width,
                delta_height - pad_height)
        return ImageOps.expand(img, padding)

    def _make_img_gt_point_pair(self, index):
        """
        Function to read images and targets
        and return padded ones
        Args: index of the images in the lists
        Returns:
            padded rgb image, padded hght image, padded label
        """
        _img = Image.open(self.images[index]).convert('RGB')
        _img_padded = self._padding(_img, 512)
        _target = Image.open(self.categories[index])
        _target_padded = self._padding(_target, 512)
        # print(_target_padded.size)
        return _img_padded, _target_padded      


def split_data(base_dir, num_class=2,
        norm=False, purpose='train', batch_size=4, augment=False):
    """
    Function to load data for separate data splits
    Args:
        base_dir: base directory for all data
        num_class: number of classes
        norm: whether to perform data normalization
        purpose: train or test
        batch_size: training / test batch_size
        augment: whether to use data augmentstion for training
    Returns:
        train_loader, val_loader, test_loder: Loaders for 3 split
    """
    if augment and num_class == 2:
        train_set = RGB(base_dir, num_class, 'rev_annot_augment',
                norm, split='train_aug')
    elif augment and num_class == 17:
        train_set = RGB(base_dir, num_class, 'gt_augment',
                norm, split='train_aug')
    elif not augment and num_class == 17:
        train_set = RGB(base_dir, num_class, 'ground_truth_500',
                norm, split='train')
    else:
        train_set = RGB(base_dir, num_class, 'rev_annotations',
                norm, split='train')
    if num_class == 17:
        val_set = RGB(base_dir, num_class, 'ground_truth_500',
                norm, split='valid')
        test_set = RGB(base_dir, num_class, 'ground_truth_500',
                norm, split='test')
    else:
        val_set = RGB(base_dir, num_class, 'rev_annotations',
                norm, split='valid')
        test_set = RGB(base_dir, num_class, 'rev_annotations',
                norm, split='test')
    if purpose == 'train':
        train_loader = DataLoader(train_set, batch_size=batch_size,
                shuffle=True, num_workers=1)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,
                shuffle=False, num_workers=1)
   
    val_loader = DataLoader(val_set, batch_size=batch_size,
            shuffle=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size,
            shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader
