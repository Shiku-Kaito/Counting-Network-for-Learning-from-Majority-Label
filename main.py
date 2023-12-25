import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from get_module import get_module

def main(args):
    fix_seed(args.seed)
    args.output_path += '%s-%dclass_%s/' % (args.dataset, args.classes, args.majority_size) 
    
    fix_seed(args.seed) 
    train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader = get_module(args)
    make_folder(args)

    if args.is_evaluation == False:
        train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function)
        return
    elif args.is_evaluation == True:
        model.load_state_dict(torch.load(("%s/model/%s/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.mode, args.fold, args.seed) ,map_location=args.device))
        result_dict = eval_net(args, test_loader, model)
        return result_dict

if __name__ == '__main__':
    results_dict = {"bag_acc":[], "ins_acc":[]}
    for fold in range(5):
        parser = argparse.ArgumentParser()
        # Data selectiion
        parser.add_argument('--fold', default=fold,
                            type=int, help='fold number')
        parser.add_argument('--dataset', default='cifar10',
                            type=str, help='cifar10 or svhn or oct or path or SpokenArabicDigits')
        parser.add_argument('--classes', #書き換え
                            default=10, type=int, help="number of the sampled instnace")
        parser.add_argument('--majority_size', #書き換え
                            default="various", type=str, help="small or various or large")
        parser.add_argument('--bag_size', default=64, type=int, help="")
        parser.add_argument('--bag_num', default=400, type=int, help="")
        # Training Setup
        parser.add_argument('--num_epochs', default=1500, type=int,
                            help='number of epochs for training.')
        parser.add_argument('--device', default='cuda:0',
                            type=str, help='device')
        parser.add_argument('--batch_size', default=64,
                            type=int, help='batch size for training.')
        parser.add_argument('--seed', default=0,
                            type=int, help='seed value')
        parser.add_argument('--num_workers', default=0, type=int,
                            help='number of workers for training.')
        parser.add_argument('--lr', default=3e-4,
                            type=float, help='learning rate')
        parser.add_argument('--is_test', default=1,
                            type=int, help='1 or 0')        # test in training epoch
        parser.add_argument('--is_evaluation', default=0,   # model evaluatiion
                            type=int, help='1 or 0')                             
        # Module Selection
        parser.add_argument('--module',default='Count', 
                            type=str, help="Count or Feat_agg or Output_agg or Att_mil or Add_mil")
        parser.add_argument('--mode',default='',   # don't write
                            type=str, help="")                        
        # Save Path
        parser.add_argument('--output_path',
                            default='./result/', type=str, help="output file name")
        
        ### Module detail ####
        # Count Parameter
        parser.add_argument('--temper1', default=0.1,
                            type=float, help='softmax temper of before counting')
        parser.add_argument('--temper2', default=0.1,
                            type=float, help='softmax temper of after counting')  
        # Traditional MIL(Feat-agg and Output-agg) Paramter
        parser.add_argument('--output_agg_method', default="mean",
                            type=str, help='mean')  
        parser.add_argument('--feat_agg_method', default="max",
                            type=str, help='mean or max or p_norm or LSE')  
        parser.add_argument('--p_val',
                            default=4, type=int, help="1 or 4 or 8")
        # Additive MIL Parameter
        parser.add_argument('--add_agg_method', default="TransMIL",
                            type=str, help='Attention_MIL or TransMIL') 
        args = parser.parse_args()

        if args.is_evaluation == False:
            main(args)

        else:
            result_dict = main(args)
            results_dict["bag_acc"].append(result_dict["bag_acc"]), results_dict["ins_acc"].append(result_dict["ins_acc"])

    if args.is_evaluation == True:
        print("=====================================================================================")
        print("5 fold cross validation, Inst acc: %.4f, Bag acc: %.4f" % (np.mean(np.array(results_dict["ins_acc"])), np.mean(np.array(results_dict["bag_acc"]))))
        print("=====================================================================================")