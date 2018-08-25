import torch
# import numpy as np
# import cv2
from torch.utils.data import Dataset
from pathlib import Path
# from pycocotools.coco import COCO
# from pycocotools import mask as cocomask
import cv2
import numpy as np
import skimage.io as io
# import matplotlib.pyplot as plt
# import pylab
# import random
# import prepare_data
import os

data_path = Path('data')
data_directory = "data/"
# annotation_file_template = "{}/{}/annotation{}.json"

TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_MASKS_DIRECTORY = "data/train/masks"
VAL_IMAGES_DIRECTORY = "data/train/images"
TEST_IMAGES_DIRECTORY = "data/test/images"

class SaltDataset(Dataset):
    def __init__(self, file_names: str, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type
        ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.coco = COCO(self.file_names)
        # self.image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())

    def __len__(self):
        # if self.mode == 'valid':
        return len(self.file_names)
        # else:
        # return 2

    def __getitem__(self, idx):
        # print(self.file_names)
        # print(idx)
        # print(self.file_names[idx], len(self.file_names), idx)
        # img = self.coco.loadImgs(self.image_ids[idx])[0]
        # annotation_ids = self.coco.getAnnIds(imgIds=img['id'])
        # annotations = self.coco.loadAnns(annotation_ids)
        if self.mode == 'predict':
            img_file_name = os.path.join(TEST_IMAGES_DIRECTORY, self.file_names[idx])
            pic = load_image(img_file_name, self.mode)
            pic, _ = self.transform(pic, None)
            # pic, _ = self.transform(pic[0], None)
        else:
            img_file_name = os.path.join(TRAIN_IMAGES_DIRECTORY, self.file_names[idx])
            mask_file_name = os.path.join(TRAIN_MASKS_DIRECTORY, self.file_names[idx])
            pic = load_image(img_file_name, self.mode)
            mask = load_image(mask_file_name, 'mask')
            pic, mask = self.transform(pic, mask)
        # plot_aug(pic, mask)
        if self.problem_type == 'binary' and self.mode == 'train':
            # return to_float_tensor(pic),\
            #        torch.from_numpy(np.expand_dims(mask, 0)).float()
            return torch.from_numpy(np.expand_dims(pic, 0)).float(), \
                   torch.from_numpy(np.expand_dims(mask, 0)).float()
        elif self.problem_type == 'binary' and self.mode == 'valid':
            # return to_float_tensor(pic),\
            #        torch.from_numpy(np.expand_dims(mask, 0)).float(), idx
            return torch.from_numpy(np.expand_dims(pic, 0)).float(), \
                   torch.from_numpy(np.expand_dims(mask, 0)).float(), idx
        elif self.mode == 'predict':
            # return to_float_tensor(pic), self.file_names[idx]
            return torch.from_numpy(np.expand_dims(pic, 0)).float(), self.file_names[idx]
        else:
            # return to_float_tensor(img), torch.from_numpy(mask).long()
            return to_float_tensor(pic), to_float_tensor(mask)
        # else:
        #     return to_float_tensor(img)# , str(img_file_name)


def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def load_image(image_path, mode):
    # if mode == 'valid':
    #     image_path = os.path.join(VAL_IMAGES_DIRECTORY, img["file_name"])
    # else:
    # image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, img)
    if mode == 'mask':
        I = cv2.imread(image_path, 0)
        I = pad(I, pad_size=32)
        I = I[0]/255
    else:
        I = cv2.imread(image_path, 0)
        # arr = np.linspace(0, 100, 101)
        # mesh = np.meshgrid(arr, arr)[0]
        # I[:,:,1] = mesh
        # I[:,:,2] = mesh.transpose(1, 0)
        I = pad(I, pad_size=32)
        I = I[0]/255
    # I1 = cv2.imread(image_path)
    # print(path_, img.shape)
    return I


def pad(img, pad_size=32):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def unpad(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]
# def load_mask(annotations, img):
#     mask = np.zeros((img['height'], img['width']))
#     for i in annotations:
#         rle = cocomask.frPyObjects(i['segmentation'], img['height'], img['width'])
#         m = cocomask.decode(rle)
#         # m.shape has a shape of (300, 300, 1)
#         # so we first convert it to a shape of (300, 300)
#         m = m.reshape((img['height'], img['width']))
#         mask += m
    # path_ = "data/stage1_train_/{}/masksmask.png".format(path)
    # if mode != 'train':
    #     path_ = "data/stage2_test/{}/images/{}.png".format(path, path)
    # if not os.path.isfile(path_):
    #     print('{} was empty'.format(path_))
    # factor = prepare_data.binary_factor
    # mask = cv2.imread(str(path_))
    # kernel = np.ones((5, 5), np.uint8)
    # mask = mask.astype(np.uint8)
    # kernel = np.ones((5, 5), np.uint8)
    # dilation = cv2.dilate(mask, kernel, iterations=10)
    # mask = dilation - mask
    # seed = cv2.erode(mask[:, :, 0], kernel, iterations=1)
    # border = mask[:, :, 0] - seed
    # mask[:, :, 1] = np.zeros(seed.shape)
    # mask[:, :, 1] = seed
    # mask[:, :, 2] = np.zeros(seed.shape)
    # mask[:, :, 2] = border

    # return mask

class MapDatasetTest(Dataset):
    def __init__(self, file_names: str, to_augment=False, transform=None, mode='predict', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        # if self.mode == 'valid':
        return len(self.file_names)
        # else:
        # return 10

    def __getitem__(self, idx):

        # print(self.file_names)
        # print(idx)
        # print(self.file_names[idx], len(self.file_names), idx)
        img_file_name = self.file_names[idx]
        img = load_image_test(img_file_name)
        # mask = None
        img, mask = self.transform(img, None)
        return to_float_tensor(img), img_file_name

def load_image_test(img):
    # path_ = "data/stage1_train_/{}/images/{}.png".format(path, path)
        # path_ = "data/stage1_test/{}/images/{}.png".format(path, path)
    # path_ = "../mapping-challenge-starter-kit/data/test_images/{}".format(img)
    path_ = "../mapping-challenge-starter-kit/data/val/images/{}".format(img)
    if not os.path.isfile(path_):
        print('{} was empty'.format(path_))
    I = io.imread(path_)
    # I1 = cv2.imread(image_path)
    # print(path_, img.shape)
    return I

# def plot_aug(pic, mask):
#     fig = plt.figure(figsize=(8, 8))
#     fig.add_subplot(1, 2, 1)
#     plt.imshow(pic)
#     fig.add_subplot(1, 2, 2)
#     plt.imshow(mask)
#     # plt.imshow(pic)
#     # plt.imshow(mask)
#     plt.show()

