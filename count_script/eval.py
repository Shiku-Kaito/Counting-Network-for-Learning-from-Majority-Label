import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import logging

from utils import *

def eval_net(args, test_loader, model):
    fix_seed(args.seed)
    result_dict = {}

    s_time = time()
    model.eval()
    ins_gt, bag_gt, ins_pred, bag_pred = [], [], [], []
    with torch.no_grad():
        for iteration, data in enumerate(test_loader): #enumerate(tqdm(test_loader, leave=False)):
            bag_label_copy=data["bag_label"].cpu().detach()
            ins_label, bag_label = data["ins_label"].reshape(-1), torch.eye(args.classes)[data["bag_label"]]
            bags, bag_label = data["bags"].to(args.device), bag_label.to(args.device)

            y = model(bags)

            ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label_copy.cpu().detach().numpy())
            ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())

    ins_gt, bag_gt, ins_pred, bag_pred = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred)
    result_dict["ins_acc"], result_dict["bag_acc"] = (ins_gt == ins_pred).mean(), (bag_gt == bag_pred).mean() 

    print("Inst acc: %.4f, Bag acc: %.4f" % (result_dict["ins_acc"], result_dict["bag_acc"]))

    return result_dict
        
