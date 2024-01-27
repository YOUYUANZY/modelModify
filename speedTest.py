import os
import time

import numpy as np
import torch
from PIL import Image

from mtcnn import detect_faces
from nets.arcface import Arcface
from nets.facenet import Facenet
from utils.utils import resize_image, preprocess_input, cvtColor


def speedTest(config):
    if config.model == 'facenet':
        FR = Facenet(backbone=config.backbone, attention=config.attention, mode='predict')
    elif config.model == 'arcface':
        FR = Arcface(backbone=config.backbone, mode='predict')
    else:
        raise ValueError('modelSummary error,unsupported model')
    FR.eval()
    device = torch.device('cuda:0')
    FR = FR.to(device)
    FR.cuda()
    start = time.time()

    files = os.listdir(config.dataPath)
    files = sorted(files)
    for i, file in enumerate(files):
        pPath = os.path.join(config.dataPath, file)
        if not os.path.isdir(pPath):
            continue
        pNames = os.listdir(pPath)
        for pName in pNames:
            if pName.endswith(config.dataType):
                img = Image.open(os.path.join(os.path.abspath(config.dataPath), file, pName))
                img = cvtColor(img)
                bound, _ = detect_faces(img)
                img = img.crop((bound[0][0], bound[0][1], bound[0][2], bound[0][3]))
                with torch.no_grad():
                    img = resize_image(img, [config.inputSize[1], config.inputSize[0]],
                                       letterbox_image=True)
                    photo = torch.from_numpy(
                        np.expand_dims(np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1)), 0))
                    photo = photo.cuda()
                    output = FR(photo).cpu().numpy()

    end = time.time()
    print(f"代码执行时间：{end-start} 秒")
