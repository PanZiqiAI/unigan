
import os
import struct
import numpy as np


########################################################################################################################
# Decoding functions
########################################################################################################################

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    :param idx3_ubyte_file:
    :return:
    """
    bin_data = open(idx3_ubyte_file, 'rb').read()
    # File header
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # Dataset
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    :param idx1_ubyte_file:
    :return:
    """
    # Load bin data
    bin_data = open(idx1_ubyte_file, 'rb').read()
    # File header
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # Dataset
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


########################################################################################################################
# Image folder
########################################################################################################################

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root_dir, max_dataset_size=float("inf")):
    assert os.path.isdir(root_dir), '%s is not a valid directory' % root_dir
    # 1. Init
    images = []
    # 2. Prepare
    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

