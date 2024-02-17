import os
import random

import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from mtcnn import detect_faces
from .utils import cvtColor, preprocess_input, resize_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class FacenetDataset(Dataset):
    def __init__(self, input_shape, lines, num_classes, random):
        self.input_shape = input_shape
        self.lines = lines
        self.length = len(lines)
        self.num_classes = num_classes
        self.random = random

        self.images = []

        # 路径和标签
        self.paths = []
        self.labels = []

        self.load_dataset()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 创建全为零的矩阵
        images = np.zeros((3, 3, self.input_shape[0], self.input_shape[1]))
        labels = np.zeros((3))
        imgs = []

        # 先获得两张同一个人的人脸
        # 用来作为anchor和positive
        while True:
            c = random.randint(0, self.num_classes - 1)
            selected = self.images[self.labels[:] == c]
            if len(selected) > 2:
                break
        image_indexes = np.random.choice(range(0, len(selected)), 2)
        imgs.append(selected[image_indexes[0]])
        imgs.append(selected[image_indexes[1]])
        # 取出另外一个人的人脸
        different_c = list(range(self.num_classes))
        different_c.pop(c)
        while True:
            different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
            if different_c_index == c:
                continue
            current_c = different_c[different_c_index[0]]
            selected = self.images[self.labels == current_c]
            if len(selected) >= 1:
                break
        image_indexes = np.random.choice(range(0, len(selected)), 1)
        imgs.append(selected[image_indexes[0]])
        if self.random:
            for i in range(3):
                if self.rand() < .5:
                    imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
        imgs = resize_image(imgs, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
        imgs_ = preprocess_input(imgs)
        for i, img in enumerate(imgs_):
            images[i, :, :, :] = np.transpose(imgs_[i], [2, 0, 1])
        labels[0] = c
        labels[1] = c
        labels[2] = current_c
        return images, labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def load_dataset(self):
        for path in tqdm(self.lines):
            path_split = path.split(";")
            img = cvtColor(Image.open(path_split[1].split()[0]))
            bound, _ = detect_faces(img)
            if len(bound) == 0:
                continue
            img = img.crop((bound[0][0], bound[0][1], bound[0][2], bound[0][3]))
            self.images.append(img)
            self.labels.append(int(path_split[0]))
        self.images = np.array(self.images, dtype=object)
        self.labels = np.array(self.labels)


class arcFaceDataset(Dataset):
    def __init__(self, input_shape, lines, random):
        self.input_shape = input_shape
        self.lines = lines
        self.random = random

    def __len__(self):
        return len(self.lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        annotation_path = self.lines[index].split(';')[1].split()[0]
        y = int(self.lines[index].split(';')[0])
        imgs = []

        imgs.append(cvtColor(Image.open(annotation_path)))
        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        if self.rand() < .5 and self.random:
            imgs[0] = imgs[0].transpose(Image.FLIP_LEFT_RIGHT)
        imgs = resize_image(imgs, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)

        image = np.transpose(preprocess_input(np.array(imgs[0], dtype='float32')), (2, 0, 1))
        return image, y


# DataLoader中collate_fn使用
# 将三张图片合为一张，便于按图片分类训练
def face_dataset_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    images1 = np.array(images)[:, 0, :, :, :]
    images2 = np.array(images)[:, 1, :, :, :]
    images3 = np.array(images)[:, 2, :, :, :]
    images = np.concatenate([images1, images2, images3], 0)

    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels3 = np.array(labels)[:, 2]
    labels = np.concatenate([labels1, labels2, labels3], 0)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).long()
    return images, labels


def arc_dataset_collate(batch):
    images = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    targets = torch.from_numpy(np.array(targets)).long()
    return images, targets


# LFW评估用数据集加载器
class LFWDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, image_size, transform=None):
        super(LFWDataset, self).__init__(dir, transform)
        self.image_size = image_size
        self.pairs_path = pairs_path
        self.validation_images = self.get_lfw_imgs(dir)

    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_imgs(self, lfw_dir, file_ext="jpg"):
        pairs = self.read_lfw_pairs(self.pairs_path)
        nrof_skipped_pairs = 0
        img_list = []
        issame_list = []
        for i in tqdm(range(len(pairs))):
            img1 = None
            img2 = None
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                img1 = Image.open(path0)
                path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                img2 = Image.open(path1)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                img1 = Image.open(path0)
                path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                img2 = Image.open(path1)
                issame = False
            else:
                raise ValueError('lfw dataloader error')
            if os.path.exists(path0) and os.path.exists(path1):
                bound1, _ = detect_faces(img1)
                if len(bound1) == 0:
                    continue
                img1 = img1.crop((bound1[0][0], bound1[0][1], bound1[0][2], bound1[0][3]))
                bound2, _ = detect_faces(img2)
                if len(bound2) == 0:
                    continue
                img2 = img2.crop((bound2[0][0], bound2[0][1], bound2[0][2], bound2[0][3]))
                imgs = [img1, img2]
                imgs = resize_image(imgs, [self.image_size[1], self.image_size[0]], letterbox_image=True)
                img_list.append((imgs, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)
        return img_list

    def __getitem__(self, index):
        (imgs, issame) = self.validation_images[index]

        image1, image2 = np.transpose(preprocess_input(np.array(imgs[0], np.float32)), [2, 0, 1]), np.transpose(
            preprocess_input(np.array(imgs[1], np.float32)), [2, 0, 1])

        return image1, image2, issame

    def __len__(self):
        return len(self.validation_images)
