import argparse
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from dataloader  import *
from torchvision.models import resnet18, resnet34, resnet50

from count_script.count_train import train_net as Count_train_net
from count_script.eval import eval_net as Count_eval_net
from count_script.count_network import Count, Count_1d

from attentionmil_script.attentionMIL_train import train_net as att_train_net
from attentionmil_script.eval import eval_net as att_eval_net
from attentionmil_script.attentionMIL_network import Attention, Attention_1d

from additive_script.additiveMIL_train import train_net as add_train_net
from additive_script.eval import eval_net as add_eval_net
from additive_script.additiveMIL_network import get_additive_mil_model_n_weights, get_transmil_model_n_weights, get_additive_transmil_model_n_weights, get_additive_mil_model_n_weights_1d
from additive_script.additive_modules.additive_transmil import  AdditiveTransMIL_1d

from MIL_script.MIL_train import train_net as MIL_train_net
from MIL_script.eval import eval_net as MIL_eval_net
from MIL_script.MIL_network import Feat_agg, Output_agg, Feat_agg_1d, Output_agg_1d

from losses import cross_entropy_loss

def get_module(args):
    if args.module ==  "Count":
        args.mode = "count_T1=%s_T2=%s" % (str(args.temper1), str(args.temper2))
        # Dataloader
        if args.dataset == "SpokenArabicDigits":
            train_loader, val_loader, test_loader = load_data_bags_1d(args)
        else: 
            train_loader, val_loader, test_loader = load_data_bags(args)        
        # Model
        if args.dataset == "SpokenArabicDigits":
            model = Count_1d(args.classes, args.temper1, args.temper2)
        else:
            model = Count(args.classes, args.temper1, args.temper2)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = cross_entropy_loss      #non softmax crossentropy 
        # Train net
        train_net = Count_train_net
        eval_net = Count_eval_net


    elif args.module ==  "Feat_agg":
        args.mode =  "Feat_%s" % args.feat_agg_method
        if args.feat_agg_method == "p_norm" or args.feat_agg_method == "LSE":
            args.mode += "=%s" % str(args.p_val)    
        # Dataloader
        if args.dataset == "SpokenArabicDigits":
            train_loader, val_loader, test_loader = load_data_bags_1d(args)
        else: 
            train_loader, val_loader, test_loader = load_data_bags(args) 
        # Model
        if args.dataset == "SpokenArabicDigits":
            model = Feat_agg_1d(args.classes, args.feat_agg_method, args.p_val) 
        else:
            model = Feat_agg(args.classes, args.feat_agg_method, args.p_val) 
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()  
        # Train net
        train_net = MIL_train_net
        eval_net = MIL_eval_net


    elif args.module ==  "Output_agg":
        args.mode = "Output_%s" % args.output_agg_method
        # Dataloader
        if args.dataset == "SpokenArabicDigits":
            train_loader, val_loader, test_loader = load_data_bags_1d(args)
        else: 
            train_loader, val_loader, test_loader = load_data_bags(args) 
        # Model
        if args.dataset == "SpokenArabicDigits":
            model = Output_agg_1d(args.classes, args.output_agg_method)
        else:
            model = Output_agg(args.classes, args.output_agg_method)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()
        # Train net
        train_net = MIL_train_net
        eval_net = MIL_eval_net

    elif args.module ==  "Att_mil":
        args.mode = "attention_mil"
        # Dataloader
        if args.dataset == "SpokenArabicDigits":
            train_loader, val_loader, test_loader = load_data_bags_1d(args)
        else: 
            train_loader, val_loader, test_loader = load_data_bags(args) 
        # Model
        if args.dataset == "SpokenArabicDigits":
            model = Attention_1d(args.classes)
        else:
            model = Attention(args.classes)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()
        # Train net
        train_net = att_train_net
        eval_net = att_eval_net

    elif args.module ==  "Add_mil":
        # Dataloader
        if args.dataset == "SpokenArabicDigits":
            train_loader, val_loader, test_loader = load_data_bags_1d(args)
        else: 
            train_loader, val_loader, test_loader = load_data_bags(args) 
        # Model
        if args.add_agg_method == "Attention_MIL":
            args.mode = "additive_mil"
            if args.dataset == "SpokenArabicDigits":
                model = get_additive_mil_model_n_weights_1d(args.classes)[0]
            else:
                model = get_additive_mil_model_n_weights(args.classes)[0]
        elif args.add_agg_method == "TransMIL":
            args.mode = "additiveTransMIL"
            if args.dataset == "SpokenArabicDigits":
                model = AdditiveTransMIL_1d(n_classes=args.classes, additive_hidden_dims=[256])
            else:
                model = get_additive_transmil_model_n_weights(args.classes)[0]
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.CrossEntropyLoss()
        # Train net
        train_net = add_train_net
        eval_net = add_eval_net

    else:
        print("Module ERROR!!!!!")

    return train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader
