import datetime
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn

from mtcnn import detect_faces
from nets.facenet import Facenet
from utils.utils import resize_image, preprocess_input, cvtColor


def speedTest(config):
    FR = Facenet(backbone=config.backbone, mode='predict')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FR.eval()
    FR.load_state_dict(torch.load(config.weightPath, map_location=device), strict=False)
    FR = torch.nn.DataParallel(FR)
    cudnn.benchmark = True
    FR = FR.cuda()

    imgs = []
    totalTime = 0
    count = 0
    files = os.listdir(config.dataPath)
    files = sorted(files)
    for i, file in enumerate(files):
        pPath = os.path.join(config.dataPath, file)
        if not os.path.isdir(pPath):
            continue
        pNames = os.listdir(pPath)
        for pName in pNames:
            if pName.endswith(config.dataType):
                path = os.path.join(os.path.abspath(config.dataPath), file, pName)
                img = cv2.imread(path)
                cv2.imshow("src", img)
                cv2.waitKey(1)
                count = count + 1
                start = time.clock()
                img = Image.open(path)
                img = cvtColor(img)
                bound, _ = detect_faces(img)
                img = img.crop((bound[0][0], bound[0][1], bound[0][2], bound[0][3]))
                with torch.no_grad():
                    img = resize_image(img, [config.inputSize[1], config.inputSize[0]],
                                       letterbox_image=True)
                    photo = torch.from_numpy(
                        np.expand_dims(np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1)), 0))
                    photo = photo.cuda()
                    end = time.clock()
                    totalTime = totalTime + end - start
                    imgs.append(photo)
    print(f"图片预处理总时间：{totalTime} s")
    print(f"图片预处理数量：{count}")
    print(f"平均图片预处理时间：{totalTime / count * 1000} ms")
    print('================================================')

    totalTime = 0
    count = 0
    for i, img in enumerate(imgs):
        count = count + 1
        start = time.time()
        output = FR(img).cpu().detach().numpy()
        end = time.time()
        totalTime = totalTime + end - start
    print(f"time.time()特征提取总时间：{totalTime} s")
    print(f"time.time()特征提取数量：{count}")
    print(f"time.time()平均特征提取时间：{totalTime / count * 1000} ms")
    print('================================================')

    totalTime = 0
    count = 0
    for i, img in enumerate(imgs):
        count = count + 1
        start = time.clock()
        output = FR(img).cpu().detach().numpy()
        end = time.clock()
        totalTime = totalTime + end - start
    print(f"time.clock()特征提取总时间：{totalTime} s")
    print(f"time.clock()特征提取数量：{count}")
    print(f"time.clock()平均特征提取时间：{totalTime / count * 1000} ms")
    print('================================================')

    totalTime = 0
    count = 0
    for i, img in enumerate(imgs):
        count = count + 1
        start = time.perf_counter()
        output = FR(img).cpu().detach().numpy()
        end = time.perf_counter()
        totalTime = totalTime + end - start
    print(f"time.perf_counter()特征提取总时间：{totalTime} s")
    print(f"time.perf_counter()特征提取数量：{count}")
    print(f"time.perf_counter()平均特征提取时间：{totalTime / count * 1000} ms")
    print('================================================')

    totalTime = 0
    count = 0
    for i, img in enumerate(imgs):
        count = count + 1
        start = datetime.datetime.now()
        output = FR(img).cpu().detach().numpy()
        end = datetime.datetime.now()
        totalTime = totalTime + (end - start).total_seconds()
    print(f"datetime.datetime.now()特征提取总时间：{totalTime} s")
    print(f"datetime.datetime.now()特征提取数量：{count}")
    print(f"datetime.datetime.now()平均特征提取时间：{totalTime / count * 1000} ms")
    print('================================================')
