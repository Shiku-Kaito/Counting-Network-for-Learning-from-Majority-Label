import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch.nn.functional as F
from statistics import mean
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from skimage import io
import glob


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

def make_dataset_folder(path):
    p = ''
    for x in path.split('/'):
        p += x+'/'
        if not os.path.exists(p):
            os.mkdir(p)

def make_folder(args):
    if not os.path.exists(args.output_path + "/acc_graph"):
        os.mkdir(args.output_path + "/acc_graph")
    if not os.path.exists(args.output_path + "/cm"):
        os.mkdir(args.output_path + "/cm")
    if not os.path.exists(args.output_path + "/log_dict"):
        os.mkdir(args.output_path + "/log_dict")
    if not os.path.exists(args.output_path + "/loss_graph"):
        os.mkdir(args.output_path + "/loss_graph")
    if not os.path.exists(args.output_path + "/model"):
        os.mkdir(args.output_path + "/model")

    if not os.path.exists(args.output_path + "/acc_graph/" + args.mode):
        os.mkdir(args.output_path + "/acc_graph/" + args.mode)
    if not os.path.exists(args.output_path + "/cm/" + args.mode):
        os.mkdir(args.output_path + "/cm/" + args.mode)
    if not os.path.exists(args.output_path + "/log_dict/" + args.mode):
        os.mkdir(args.output_path + "/log_dict/" + args.mode)
    if not os.path.exists(args.output_path + "/loss_graph/" + args.mode):
        os.mkdir(args.output_path + "/loss_graph/" + args.mode)
    if not os.path.exists(args.output_path + "/model/" + args.mode):
        os.mkdir(args.output_path + "/model/" + args.mode)
    return

def save_confusion_matrix(cm, path, title=''):
    plt.figure()
    cm = cm / cm.sum(axis=-1, keepdims=1)
    sns.heatmap(cm, annot=True, cmap='Blues_r', fmt='.2f')
    plt.xlabel('pred')
    plt.ylabel('GT')
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def cal_OP_PC_mIoU(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    FN_c = np.zeros(num_classes)
    for i in range(num_classes):
        FN_c[i] = cm[:, i].sum()-cm[i][i]

    OP = TP_c.sum() / (TP_c+FP_c).sum()
    PC = (TP_c/(TP_c+FP_c)).mean()
    mIoU = (TP_c/(TP_c+FP_c+FN_c)).mean()

    return OP, PC, mIoU


def cal_mIoU(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    FN_c = np.zeros(num_classes)
    for i in range(num_classes):
        FN_c[i] = cm[:, i].sum()-cm[i][i]

    mIoU = (TP_c/(TP_c+FP_c+FN_c)).mean()

    return mIoU

    
def make_loss_graph(args, keep_train_loss, keep_valid_loss, path):
    #loss graph save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(keep_train_loss, label = 'train')
    ax.plot(keep_valid_loss, label = 'valid')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("Losses")
    plt.legend()
    fig.savefig(path)
    plt.close() 
    return

def make_bag_acc_graph(args, train_major_acc, val_major_acc, path):
    #Bag level accuracy save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_major_acc, label = 'train bag acc')
    ax.plot(val_major_acc, label = 'valid bag acc')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("accuracy")
    plt.legend()
    fig.savefig(path)
    plt.close()
    return

def make_ins_acc_graph(args, train_ins_acc, val_ins_acc, path):
    #instance level accuracy save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_ins_acc, label = 'train instance acc')
    ax.plot(val_ins_acc, label = 'valid instans acc')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("accuracy")
    plt.legend()
    fig.savefig(path)
    plt.close()
    return

