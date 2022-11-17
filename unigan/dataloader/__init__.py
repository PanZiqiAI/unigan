
import torch
from torch.utils.data.dataloader import DataLoader
from custom_pkg.pytorch.operations import DataCycle
from torchvision import transforms as torch_transforms
from dataloader.datasets import CelebAHQ, AFHQ, FFHQ, LSUN, ImageMNIST, get_dataset_without_labels_given_categories


# ----------------------------------------------------------------------------------------------------------------------
# Dequantization transforms
# ----------------------------------------------------------------------------------------------------------------------

class Dequant(object):
    """
    Dequantize an image (C, H, W) in the range [0.0, 1.0].
    """
    def __init__(self, n_bits):
        assert n_bits <= 8
        # Config
        self._n_bits = n_bits
        self._n_bins = 2**self._n_bits

    def __call__(self, x):
        # Prepare image.
        x = x * 255.0
        if self._n_bits < 8: x = torch.floor(x / 2**(8 - self._n_bits))
        # --------------------------------------------------------------------------------------------------------------
        # Dequantization
        # --------------------------------------------------------------------------------------------------------------
        # 1. Normalize.
        x = x / self._n_bins - 0.5
        # 2. Adding noise.
        x = x + torch.rand_like(x) / self._n_bins
        # Return
        return x * 2.0


# ----------------------------------------------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------------------------------------------

class DequantCelebAHQ(CelebAHQ):
    """
    Dequantized CelebA-HQ dataset.
    """
    def __init__(self, image_size, image_n_bits, maxsize):
        super(DequantCelebAHQ, self).__init__(
            image_size=image_size, max_dataset_size=maxsize, transforms=torch_transforms.Compose([
                torch_transforms.CenterCrop(image_size), torch_transforms.RandomHorizontalFlip(),
                torch_transforms.ToTensor(), Dequant(image_n_bits)]))


class DequantAFHQ(AFHQ):
    """
    Dequantized AFHQ dataset.
    """
    def __init__(self, category, image_size, image_n_bits, maxsize):
        super(DequantAFHQ, self).__init__(
            image_size=image_size, category=category, max_dataset_size=maxsize, transforms=torch_transforms.Compose([
                torch_transforms.CenterCrop(image_size), torch_transforms.RandomHorizontalFlip(),
                torch_transforms.ToTensor(), Dequant(image_n_bits)]))


class DequantFFHQ(FFHQ):
    """
    Dequantized FFHQ dataset.
    """
    def __init__(self, image_size, image_n_bits, maxsize):
        super(DequantFFHQ, self).__init__(
           image_size=image_size, max_dataset_size=maxsize, transforms=torch_transforms.Compose([
                torch_transforms.CenterCrop(image_size), torch_transforms.RandomHorizontalFlip(),
                torch_transforms.ToTensor(), Dequant(image_n_bits)]))


class DequantLSUN(LSUN):
    """
    Dequantized LSUN dataset.
    """
    def __init__(self, selection, image_size, image_n_bits, maxsize):
        selection = {
            'bedroom': 'scenes/bedroom/train', 'church': 'scenes/church_outdoor/train',
            'cat': 'objects/cat', 'car': 'objects/car', 'airplane': 'objects/airplane'}[selection]
        super(DequantLSUN, self).__init__(
            selection=selection, image_size=image_size, max_dataset_size=maxsize, transforms=torch_transforms.Compose([
                torch_transforms.CenterCrop(image_size), torch_transforms.RandomHorizontalFlip(),
                torch_transforms.ToTensor(), Dequant(image_n_bits)]))


def generate_data(cfg, **kwargs):
    # 1. Get dataset.
    # ------------------------------------------------------------------------------------------------------------------
    # Simple datasets.
    # ------------------------------------------------------------------------------------------------------------------
    if cfg.args.dataset == 'single-mnist':
        dataset = get_dataset_without_labels_given_categories(ImageMNIST, categories=[cfg.args.dataset_category], phase='train', to32x32=True)
    elif cfg.args.dataset == 'mnist':
        dataset = ImageMNIST(phase='train', to32x32=True)
    # ------------------------------------------------------------------------------------------------------------------
    # Real datasets.
    # ------------------------------------------------------------------------------------------------------------------
    elif cfg.args.dataset == 'celeba-hq':
        dataset = DequantCelebAHQ(image_size=cfg.args.img_size, image_n_bits=cfg.args.img_n_bits, maxsize=cfg.args.dataset_maxsize)
    elif cfg.args.dataset == 'afhq':
        dataset = DequantAFHQ(category=cfg.args.dataset_category, image_size=cfg.args.img_size, image_n_bits=cfg.args.img_n_bits, maxsize=cfg.args.dataset_maxsize)
    elif cfg.args.dataset == 'ffhq':
        dataset = DequantFFHQ(image_size=cfg.args.img_size, image_n_bits=cfg.args.img_n_bits, maxsize=cfg.args.dataset_maxsize)
    elif cfg.args.dataset.startswith('lsun'):
        dataset = DequantLSUN(selection=cfg.args.dataset.split("@")[1], image_size=cfg.args.img_size, image_n_bits=cfg.args.img_n_bits, maxsize=cfg.args.dataset_maxsize)
    else:
        raise NotImplementedError
    # 2. Get dataloader
    dataloader = DataLoader(
        dataset, batch_size=cfg.args.batch_size if 'batch_size' not in kwargs.keys() else kwargs['batch_size'],
        drop_last=cfg.args.dataset_drop_last, shuffle=cfg.args.dataset_shuffle, num_workers=cfg.args.dataset_num_threads)
    # Return
    return DataCycle(dataloader)
