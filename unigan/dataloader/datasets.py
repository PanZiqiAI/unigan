
import os
import copy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from dataloader.transforms import Flatten, To32x32
from torchvision import transforms as torch_transforms
from torchvision.datasets.folder import default_loader
from dataloader.utils import decode_idx1_ubyte, decode_idx3_ubyte, make_dataset


DATA_ROOT = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../../Datasets")


########################################################################################################################
# Classification
########################################################################################################################

class BaseClassification(Dataset):
    """
    Base class for dataset for classification.
    """
    _data = None
    _label: np.ndarray      # Each element is a integer indicating the category index.

    def __init__(self):
        # 1. Samples per category
        self._sample_indices = [np.argwhere(self._label == y)[:, 0].tolist() for y in range(self.num_classes)]
        # 2. Class counter
        self._class_counter = [len(samples) for samples in self._sample_indices]

    @property
    def num_classes(self):
        """
        :return: A integer indicating number of categories.
        """
        return len(set(self._label))

    @property
    def class_counter(self):
        """
        :return: A list, whose i-th element equals to the total sample number of the i-th category.
        """
        return self._class_counter

    @property
    def sample_indices(self):
        """
        :return: A list, whose i-th element is a numpy.array containing sample indices of the i-th category.
        """
        return self._sample_indices

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        """
        Should return x, y, where y is the class label.
        :param index:
        :return:
        """
        raise NotImplementedError

    def subset(self, categories):
        # 1. Create an instance
        dataset = copy.deepcopy(self)
        class_idx_to_orig = {}
        # 2. Modify
        dataset._data, dataset._label = [], []
        for cat_index, y in enumerate(categories):
            # Data & label
            for index in self._sample_indices[y]:
                dataset._data.append(self._data[index])
                dataset._label.append(cat_index)
            # Idx
            class_idx_to_orig[cat_index] = y
        # (3) Get label
        dataset._label = np.array(dataset._label, dtype=np.long)
        # Initialize
        BaseClassification.__init__(dataset)
        setattr(dataset, 'class_idx_to_orig', class_idx_to_orig)
        # Return
        return dataset


def get_dataset_without_labels_given_categories(base_class, categories, *args, **kwargs):
    """
    :param base_class:
    :param categories: List of integers.
    :param args:
    :param kwargs:
    :return:
    """

    class _DatasetWithoutLabels(base_class):
        """
        Reimplement __getitem__.
        """
        def __getitem__(self, item):
            _x, _y = super(_DatasetWithoutLabels, self).__getitem__(item)
            return _x

    return _DatasetWithoutLabels(*args, **kwargs).subset(categories=categories)


# ----------------------------------------------------------------------------------------------------------------------
# MNIST & FashionMNIST
# ----------------------------------------------------------------------------------------------------------------------

def mnist_paths(name):
    assert name in ['mnist', 'fashion_mnist']
    return {
        'train': (os.path.join(DATA_ROOT, "%s/train-images.idx3-ubyte" % name),
                  os.path.join(DATA_ROOT, "%s/train-labels.idx1-ubyte" % name)),
        'test': (os.path.join(DATA_ROOT, "%s/t10k-images.idx3-ubyte" % name),
                 os.path.join(DATA_ROOT, "%s/t10k-labels.idx1-ubyte" % name))}


class MNIST(BaseClassification):
    """
    MNIST dataset.
    """
    def __init__(self, images_path, labels_path, transforms=None):
        # Data & label
        self._data = decode_idx3_ubyte(images_path).astype('float32')[:, :, :, np.newaxis] / 255.0
        self._label = decode_idx1_ubyte(labels_path).astype('int64')
        # Transforms
        self._transforms = transforms
        # Initialize
        super(MNIST, self).__init__()

    def __getitem__(self, index):
        # 1. Get image & label
        image, label = self._data[index], self._label[index]
        # 2. Transform
        if self._transforms:
            image = self._transforms(image)
        # Return
        return image, label


class FlattenMNIST(MNIST):
    """
    Flatten MNIST dataset.
    """
    def __init__(self, phase, name='mnist', **kwargs):
        assert phase in ['train', 'test']
        # Get transforms
        transforms = torch_transforms.Compose([
            torch_transforms.ToTensor(), torch_transforms.Normalize((0.5, ), (0.5, )),
            Flatten()]
        ) if 'transforms' not in kwargs.keys() else kwargs['transforms']
        # Init
        super(FlattenMNIST, self).__init__(*(mnist_paths(name)[phase]), transforms=transforms)


class ImageMNIST(MNIST):
    """
    Flatten MNIST dataset.
    """
    def __init__(self, phase, name='mnist', **kwargs):
        assert phase in ['train', 'test']
        # Get transforms
        if 'transforms' in kwargs.keys():
            transforms = kwargs['transforms']
        else:
            transforms = [torch_transforms.ToTensor(), torch_transforms.Normalize((0.5, ), (0.5, ))]
            if 'to32x32' in kwargs.keys() and kwargs['to32x32']: transforms = [To32x32()] + transforms
            transforms = torch_transforms.Compose(transforms)
        # Init
        super(ImageMNIST, self).__init__(*(mnist_paths(name)[phase]), transforms=transforms)


########################################################################################################################
# Images.
########################################################################################################################

def default_transforms(**kwargs):
    # 1. Init result
    transform_list = []
    # 2. Transforms
    # (1) To grayscale
    try:
        if kwargs['grayscale']: transform_list.append(torch_transforms.Grayscale(1))
    except KeyError:
        pass
    # (2) Resize or scale
    if 'load_size' in kwargs.keys():
        transform_list.append(torch_transforms.Resize(kwargs['load_size'], Image.BICUBIC))
    # (3) Crop
    if 'crop_size' in kwargs.keys():
        # Get crop type
        crop_func = torch_transforms.RandomCrop
        if 'crop_type' in kwargs.keys():
            crop_func = {
                'random': torch_transforms.RandomCrop,
                'center': torch_transforms.CenterCrop
            }[kwargs['crop_type']]
        # Save
        transform_list.append(crop_func(kwargs['crop_size']))
    # (4) Flip
    try:
        if kwargs['flip']: transform_list.append(torch_transforms.RandomHorizontalFlip())
    except KeyError:
        pass
    # (5) To tensor
    try:
        if kwargs['to_tensor']: transform_list += [torch_transforms.ToTensor()]
    except KeyError:
        pass
    # (6) Normalize
    if 'normalize' in kwargs.keys():
        # Get normalize
        if kwargs['normalize'] == 'default':
            if 'grayscale' in kwargs.keys() and kwargs['grayscale']: norm = torch_transforms.Normalize((0.5, ), (0.5, ))
            else: norm = torch_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            norm = torch_transforms.Normalize(*kwargs['normalize'])
        # Save
        transform_list.append(norm)
    # Return
    return torch_transforms.Compose(transform_list)


def default_paired_transforms(grayscales, **kwargs):
    return [
        # Transforms a
        default_transforms(grayscale=grayscales[0], **kwargs),
        # Transforms b
        default_transforms(grayscale=grayscales[1], **kwargs)
    ]


class SingleImageDataset(Dataset):
    """
        A dataset class for single image dataset.
        It assumes that the root_dir contains images.
    """
    def __init__(self, root_dir, transforms, max_dataset_size=float("inf")):
        super(SingleImageDataset, self).__init__()
        # Path
        self._paths = sorted(make_dataset(root_dir, max_dataset_size))
        # Transforms
        self._transforms = transforms

    def __getitem__(self, index):
        """
        :return: An image.
        """
        # 1. Load
        img = default_loader(self._paths[index])
        # 2. Transform
        img = self._transforms(img)
        # Return
        return img

    def __len__(self):
        return len(self._paths)


# ----------------------------------------------------------------------------------------------------------------------
# celeba
# ----------------------------------------------------------------------------------------------------------------------

class CelebAHQ(SingleImageDataset):
    """
    CelebA-HQ dataset.
    """
    def __init__(self, image_size, max_dataset_size=float("inf"), **kwargs):
        super(CelebAHQ, self).__init__(
            root_dir=os.path.join(DATA_ROOT, 'celeba-hq/images%dx%d' % (image_size, image_size)), max_dataset_size=max_dataset_size,
            transforms=default_transforms(**kwargs, to_tensor=True, normalize='default') 
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


# ----------------------------------------------------------------------------------------------------------------------
# AFHQ
# ----------------------------------------------------------------------------------------------------------------------

class AFHQ(SingleImageDataset):
    """
    AFHQ dataset.
    """
    def __init__(self, image_size, category, phase='train', max_dataset_size=float("inf"), **kwargs):
        super(AFHQ, self).__init__(
            root_dir=os.path.join(DATA_ROOT, 'afhq/images%dx%d' % (image_size, image_size), phase, category), max_dataset_size=max_dataset_size,
            transforms=default_transforms(**kwargs, to_tensor=True, normalize='default')
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


# ----------------------------------------------------------------------------------------------------------------------
# FFHQ
# ----------------------------------------------------------------------------------------------------------------------

class FFHQ(SingleImageDataset):
    """
    FFHQ dataset.
    """
    def __init__(self, image_size, max_dataset_size=float("inf"), **kwargs):
        super(FFHQ, self).__init__(
            root_dir=os.path.join(DATA_ROOT, 'ffhq/images%dx%d' % (image_size, image_size)), max_dataset_size=max_dataset_size,
            transforms=default_transforms(**kwargs, to_tensor=True, normalize='default')
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])


# ----------------------------------------------------------------------------------------------------------------------
# LSUN
# ----------------------------------------------------------------------------------------------------------------------

class LSUN(SingleImageDataset):
    """
    LSUN dataset.
    """
    def __init__(self, selection, image_size=None, max_dataset_size=float("inf"), **kwargs):
        super(LSUN, self).__init__(
            root_dir=os.path.join(DATA_ROOT, 'lsun', 'images' if image_size is None else 'images%dx%d' % (image_size, image_size), selection),
            max_dataset_size=max_dataset_size, transforms=default_transforms(**kwargs, to_tensor=True, normalize='default')
            if 'transforms' not in kwargs.keys() else kwargs['transforms'])
