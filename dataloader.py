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


#toy
class DatasetBag_addbag(torch.utils.data.Dataset):
    def __init__(self, args, data, ins_label, major_labels):
        np.random.seed(args.seed)
        self.data = data
        self.ins_label = ins_label
        self.major_labels = major_labels
        # self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.data)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        # choice_bags = self.data[np.random.randint(0, self.len)]
        data = self.data[idx]
        (b, w, h, c) = data.shape
        trans_data = torch.zeros((b, c, w, h))
        for i in range(b):
            trans_data[i] = self.transform(data[i])
        data = trans_data
        ins_label = self.ins_label[idx]
        ins_label = torch.tensor(ins_label).long()
        major_label = self.major_labels[idx]
        major_label = torch.tensor(major_label).long()
        return {"bags": data, "ins_label": ins_label, "bag_label": major_label}

def load_data_bags(args):  # Toy
    ######### load data #######
    test_data = np.load('./data/%s/%dclass_%s/%d/test_bags.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    test_ins_labels = np.load('./data/%s/%dclass_%s/%d/test_ins_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    test_major_labels = np.load('./data/%s/%dclass_%s/%d/test_major_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    train_bags = np.load('./data/%s/%dclass_%s/%d/train_bags.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    train_ins_labels = np.load('./data/%s/%dclass_%s/%d/train_ins_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    train_major_labels = np.load('./data/%s/%dclass_%s/%d/train_major_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    val_bags = np.load('./data/%s/%dclass_%s/%d/val_bags.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    val_ins_labels = np.load('./data/%s/%dclass_%s/%d/val_ins_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    val_major_labels = np.load('./data/%s/%dclass_%s/%d/val_major_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    train_dataset = DatasetBag_addbag(
        args=args, data=train_bags, ins_label=train_ins_labels, major_labels=train_major_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers)
    val_dataset = DatasetBag_addbag(
        args=args, data=val_bags, ins_label=val_ins_labels, major_labels=val_major_labels)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False,  num_workers=args.num_workers)
    test_dataset = DatasetBag_addbag( args=args, data=test_data, ins_label=test_ins_labels, major_labels=test_major_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False,  num_workers=args.num_workers)
    return train_loader, val_loader, test_loader


#toy
class DatasetBag_addbag_1d(torch.utils.data.Dataset):
    def __init__(self, args, data, ins_label, major_labels):
        np.random.seed(args.seed)
        self.data = data
        self.ins_label = ins_label
        self.major_labels = major_labels
        # self.augment = augment
        self.len = len(self.data)

    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        # choice_bags = self.data[np.random.randint(0, self.len)]
        data = self.data[idx].transpose(0, 2, 1)
        data = torch.tensor(data).float()
        ins_label = self.ins_label[idx]
        ins_label = torch.tensor(ins_label).long()
        major_label = self.major_labels[idx]
        major_label = torch.tensor(major_label).long()
        return {"bags": data, "ins_label": ins_label, "bag_label": major_label}

def load_data_bags_1d(args):  # Toy
    ######### load data #######
    test_data = np.load('./data/%s/%dclass_%s/%d/test_bags.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    test_ins_labels = np.load('./data/%s/%dclass_%s/%d/test_ins_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    test_major_labels = np.load('./data/%s/%dclass_%s/%d/test_major_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    train_bags = np.load('./data/%s/%dclass_%s/%d/train_bags.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    train_ins_labels = np.load('./data/%s/%dclass_%s/%d/train_ins_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    train_major_labels = np.load('./data/%s/%dclass_%s/%d/train_major_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    val_bags = np.load('./data/%s/%dclass_%s/%d/val_bags.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    val_ins_labels = np.load('./data/%s/%dclass_%s/%d/val_ins_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    val_major_labels = np.load('./data/%s/%dclass_%s/%d/val_major_labels.npy' % (args.dataset, args.classes, args.majority_size, args.fold))
    train_dataset = DatasetBag_addbag_1d(
        args=args, data=train_bags, ins_label=train_ins_labels, major_labels=train_major_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers)
    val_dataset = DatasetBag_addbag_1d(
        args=args, data=val_bags, ins_label=val_ins_labels, major_labels=val_major_labels)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False,  num_workers=args.num_workers)
    test_dataset = DatasetBag_addbag_1d( args=args, data=test_data, ins_label=test_ins_labels, major_labels=test_major_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False,  num_workers=args.num_workers)
    return train_loader, val_loader, test_loader

