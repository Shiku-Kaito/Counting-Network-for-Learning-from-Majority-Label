import  _int_paths
from utils import make_folder
from load_svhn import load_svhn
from load_cifar10 import load_cifar10
import random
import numpy as np
import matplotlib.pyplot as plt
from tslearn.datasets import UCR_UEA_datasets
from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.datasets import load_UCR_UEA_dataset
from sktime.datatypes._panel._convert import from_nested_to_2d_array
import pandas as pd


def load_SpokenArabicDigits():
    # UCR
    dataset = 'SpokenArabicDigits'

    # make_folder('./dataset/%s/' % dataset)
    print(dataset)

    data, label = [], []
    train_data, train_label = load_from_tsfile_to_dataframe(
        './data/%s/%sEq_TRAIN.ts' % (dataset, dataset)
    )
    test_data, test_label = load_from_tsfile_to_dataframe(
        './data/%s/%sEq_TEST.ts' % (dataset, dataset)
    )

    train_data = from_nested_to_2d_array(train_data)
    train_data = np.array(train_data).reshape(-1, 13, 65).transpose(0, 2, 1)
    train_label = np.array(train_label).astype(int)-1

    test_data = from_nested_to_2d_array(test_data)
    test_data = np.array(test_data).reshape(-1, 13, 65).transpose(0, 2, 1)
    test_label = np.array(test_label).astype(int)-1

    data.extend(train_data)
    data.extend(test_data)
    label.extend(train_label)
    label.extend(test_label)
    data, label = np.array(data), np.array(label)
    print(data)
    num_classes = label.max()+1
    print(data.shape, label.shape, num_classes)

    index = len(data)
    index = np.arange(index)
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label

if __name__ == '__main__':
    train_img, train_label, test_img, test_label = load_SpokenArabicDigits()
    print('==========================')
    print('Dataset information')
    print('train_img.shape, train_label.shape: ')
    print(train_img.shape, train_label.shape)
    print('test_img.shape, test_label.shape: ')
    print(test_img.shape, test_label.shape)
    # print('train_img.min(), train_img.max(): ')
    # print(train_img.min(), train_img.max())
    # print('train_label.min(), train_label.max(): ')
    # print(train_label.min(), train_label.max())
    # print(train_label)
    # print('example: ')
    # print(train_img[:5], train_label[:5])
    print('==========================')
