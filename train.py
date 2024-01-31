import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data import DataLoader

from nets.facenet import Facenet
from utils.dataloader import FacenetDataset, face_dataset_collate
from utils.epochTrain import epochTrain
from utils.lossRecord import LossHistory
from utils.training import get_Lr_Fun, set_lr, triplet_loss
from utils.utils import get_num_classes, seed_everything


def train(config):
    seed_everything(11)
    # 获取训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 一个标记(既负责提示信息又代表设备序号)
    flag = 0
    # 获取标签数量
    num_classes = get_num_classes(config.dataPath)
    # 加载模型
    model = Facenet(backbone=config.backbone, num_classes=num_classes)
    # 加载权重
    if config.weightPath != '':
        if flag == 0:
            print('Load weights {}.'.format(config.weightPath))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(config.weightPath, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # 没加载上的权重
        if flag == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # 获取损失函数
    loss = triplet_loss()
    # 记录Loss
    if flag == 0:
        loss_history = LossHistory('logs', model, input_shape=config.inputSize)
    else:
        loss_history = None
    # torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    scaler = GradScaler()
    # 启用模型训练
    model_train = model.train()
    model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.cuda()
    # 划分训练集和验证集
    with open(config.dataPath, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * config.valRate)
    num_train = len(lines) - num_val
    print(
        "num_classes: " + str(num_classes) + "\n" + "num_train: " + str(num_train) + "\n" + "num_val: " + str(num_val))
    # 自适应调整学习率
    if config.batchSize % 3 != 0 and config.model == 'facenet':
        raise ValueError("Batch_size must be the multiple of 3.")
    maxLR = min(max(config.batchSize / config.nbs * config.maxLR, config.minLimitLR), config.maxLimitLR)
    minLR = min(max(config.batchSize / config.nbs * config.minLR, config.minLimitLR * 1e-2),
                config.maxLimitLR * 1e-2)
    # 获得优化器
    optimizer = {
        'adam': optim.Adam(model.parameters(), maxLR, betas=(config.momentum, 0.999),
                           weight_decay=config.weightDecay),
        'sgd': optim.SGD(model.parameters(), maxLR, momentum=config.momentum, nesterov=True,
                         weight_decay=config.weightDecay)
    }[config.optimizer]
    # 获得学习率下降的公式
    lr_func = get_Lr_Fun(config.LrDecayType, maxLR, minLR, config.endEpoch, config.LRscheduler)
    # 判断每个轮次的批次数
    epoch_step = num_train // config.batchSize
    epoch_step_val = num_val // config.batchSize
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
    # 构建数据集加载器
    train_dataset = FacenetDataset(config.inputSize, lines[:num_train], num_classes, random=True)
    val_dataset = FacenetDataset(config.inputSize, lines[num_train:], num_classes, random=False)
    # 获得训练和验证数据集
    train_sampler = None
    val_sampler = None
    shuffle = True
    batchSize = config.batchSize // 3
    collate_fn = face_dataset_collate
    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batchSize,
                     num_workers=config.numWorkers,
                     pin_memory=True,
                     drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batchSize,
                         num_workers=config.numWorkers,
                         pin_memory=True,
                         drop_last=True, collate_fn=collate_fn, sampler=val_sampler)
    # 开始训练

    for epoch in range(config.startEpoch, config.endEpoch):
        set_lr(optimizer, lr_func, epoch)
        epochTrain(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val,
                   gen, gen_val, config.endEpoch, config.batchSize // 3,
                   scaler, config.savePeriod, 'logs', flag)
    # 训练结束
    if flag == 0:
        loss_history.writer.close()
