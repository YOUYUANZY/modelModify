import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # data = pd.read_csv('train_loss.csv')
    # plt.figure()
    # plt.xticks(np.arange(0, 60, 5))  # 设置x轴的刻度间隔
    # plt.yticks(np.arange(0, 20, 1))  # 设置y轴的刻度间隔
    # for head, data in data.iteritems():
    #     iters = range(len(data))
    #     plt.plot(iters, data.values, label=head)
    # plt.grid(True)
    # plt.xlabel('Epoch')
    # plt.ylabel('loss')
    # plt.legend(loc="upper right")
    # plt.savefig("train_loss.png")
    # plt.cla()
    # plt.close("all")
    # data = pd.read_csv('val_loss.csv')
    # plt.figure()
    # plt.xticks(np.arange(0, 60, 5))  # 设置x轴的刻度间隔
    # plt.yticks(np.arange(0, 20, 1))  # 设置y轴的刻度间隔
    # for head, data in data.iteritems():
    #     iters = range(len(data))
    #     plt.plot(iters, data.values, label=head)
    # plt.grid(True)
    # plt.xlabel('Epoch')
    # plt.ylabel('loss')
    # plt.legend(loc="upper right")
    # plt.savefig("val_loss.png")
    # plt.cla()
    # plt.close("all")
    data1 = pd.read_csv('tpr.csv')
    data2 = pd.read_csv('fpr.csv')
    plt.figure()
    for (head1, data1), (head2, data2) in zip(data1.iteritems(), data2.iteritems()):
        plt.plot(data2.values, data1.values, label=head1)
    plt.grid(True)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend(loc="upper right")
    plt.savefig("roc.png")
    plt.cla()
    plt.close("all")
