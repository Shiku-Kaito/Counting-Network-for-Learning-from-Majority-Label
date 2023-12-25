
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from utils import *


def train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function):
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("%s/log_dict/%s/fold=%d_seed=%d_training_setting.log" %  (args.output_path, args.mode, args.fold, args.seed))
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logging.info(args)
    fix_seed(args.seed)
    log_dict = {"train_bag_acc":[], "train_ins_acc":[], "train_mIoU":[], "train_loss":[], 
                "val_bag_acc":[], "val_ins_acc":[], "val_mIoU":[], "val_loss":[], 
                "test_bag_acc":[], "test_ins_acc":[], "test_mIoU":[], "test_loss":[]}

    best_val_loss = float('inf')
    cnt = 0
    ins_best_epoch=0
    for epoch in range(args.num_epochs):

        ############ train ###################
        s_time = time()
        model.train()
        ins_gt, bag_gt, ins_pred, bag_pred, losses = [], [], [], [], []
        for iteration, data in enumerate(train_loader): #enumerate(tqdm(train_loader, leave=False)):
            ins_label = data["ins_label"].reshape(-1)
            bags, bag_label = data["bags"].to(args.device), data["bag_label"].to(args.device)

            y = model(bags)
            loss = loss_function(y["bag"], bag_label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label.cpu().detach().numpy())
            ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())
            losses.append(loss.item())

        ins_gt, bag_gt, ins_pred, bag_pred = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred)
        log_dict["train_ins_acc"].append((ins_gt == ins_pred).mean()), log_dict["train_bag_acc"].append((bag_gt == bag_pred).mean()), log_dict["train_loss"].append(np.array(losses).mean())

        train_cm = confusion_matrix(y_true=ins_gt, y_pred=ins_pred, normalize='true')
        log_dict["train_mIoU"].append(cal_mIoU(train_cm))

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] train loss: %.4f, ins acc: %.4f, bag acc:  %.4f, mIoU: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["train_loss"][-1], log_dict["train_ins_acc"][-1], log_dict["train_bag_acc"][-1], log_dict["train_mIoU"][-1]))
        
        ################# validation ####################
        s_time = time()
        model.eval()
        ins_gt, bag_gt, ins_pred, bag_pred, bag_m, losses = [], [], [], [], [], []
        with torch.no_grad():
            for iteration, data in enumerate(val_loader): #enumerate(tqdm(val_loader, leave=False)):
                ins_label = data["ins_label"].reshape(-1)
                bags, bag_label = data["bags"].to(args.device), data["bag_label"].to(args.device)

                y = model(bags)
                loss = loss_function(y["bag"], bag_label)

                ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label.cpu().detach().numpy())
                ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())
                losses.append(loss.item())

        ins_gt, bag_gt, ins_pred, bag_pred = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred)
        log_dict["val_ins_acc"].append((ins_gt == ins_pred).mean()), log_dict["val_bag_acc"].append((bag_gt == bag_pred).mean()), log_dict["val_loss"].append(np.array(losses).mean())

        val_cm = confusion_matrix(y_true=ins_gt, y_pred=ins_pred, normalize='true')
        log_dict["val_mIoU"].append(cal_mIoU(val_cm))

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] val loss: %.4f, ins acc: %.4f, bag acc:  %.4f, mIoU: %.4f' %
                        (epoch+1, args.num_epochs, e_time-s_time, log_dict["val_loss"][-1], log_dict["val_ins_acc"][-1], log_dict["val_bag_acc"][-1], log_dict["val_mIoU"][-1]))

        if args.is_test == True:
            ################## test ###################
            s_time = time()
            model.eval()
            ins_gt, bag_gt, ins_pred, bag_pred = [], [], [], []
            with torch.no_grad():
                for iteration, data in enumerate(test_loader): #enumerate(tqdm(test_loader, leave=False)):
                    ins_label = data["ins_label"].reshape(-1)
                    bags, bag_label = data["bags"].to(args.device), data["bag_label"].to(args.device)

                    y = model(bags)

                    ins_gt.extend(ins_label.cpu().detach().numpy()), bag_gt.extend(bag_label.cpu().detach().numpy())
                    ins_pred.extend(y["ins"].argmax(1).cpu().detach().numpy()), bag_pred.extend(y["bag"].argmax(1).cpu().detach().numpy())

            ins_gt, bag_gt, ins_pred, bag_pred = np.array(ins_gt), np.array(bag_gt), np.array(ins_pred), np.array(bag_pred)
            log_dict["test_ins_acc"].append((ins_gt == ins_pred).mean()), log_dict["test_bag_acc"].append((bag_gt == bag_pred).mean()) 

            test_cm = confusion_matrix(y_true=ins_gt, y_pred=ins_pred, normalize='true')
            log_dict["test_mIoU"].append(cal_mIoU(test_cm))

            e_time = time()
            logging.info('[Epoch: %d/%d (%ds)] , test ins acc: %.4f, bag acc: %.4f,  mIoU: %.4f' %
                            (epoch+1, args.num_epochs, e_time-s_time, log_dict["test_ins_acc"][-1], log_dict["test_bag_acc"][-1],  log_dict["test_mIoU"][-1]))
        logging.info('===============================')

        if best_val_loss > log_dict["val_loss"][-1]:
            best_val_loss = log_dict["val_loss"][-1]
            cnt = 0
            best_epoch = epoch
            torch.save(model.state_dict(), ("%s/model/%s/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.mode, args.fold, args.seed))
            save_confusion_matrix(cm=train_cm, path=("%s/cm/%s/fold=%d_seed=%d-cm_train.png") % (args.output_path, args.mode, args.fold, args.seed),
                        title='train: epoch: %d, acc: %.4f, mIoU: %.4f' % (epoch+1, log_dict["train_ins_acc"][epoch], log_dict["train_mIoU"][epoch]))
            save_confusion_matrix(cm=val_cm, path=("%s/cm/%s/fold=%d_seed=%d-cm_val.png") % (args.output_path, args.mode, args.fold, args.seed),
                        title='validation: epoch: %d, acc: %.4f, mIoU: %.4f' % (epoch+1, log_dict["val_ins_acc"][epoch], log_dict["val_mIoU"][epoch]))
            if args.is_test == True:
                save_confusion_matrix(cm=test_cm, path=("%s/cm/%s/fold=%d_seed=%d-cm_test.png") % (args.output_path, args.mode, args.fold, args.seed),
                            title='test: epoch: %d, acc: %.4f, mIoU: %.4f' % (epoch+1, log_dict["test_ins_acc"][epoch], log_dict["test_mIoU"][epoch]))
        else:
            cnt += 1
            # if args.patience == cnt:
            #     break

        logging.info('best epoch: %d, val bag acc: %.4f, val inst acc: %.4f, mIoU: %.4f' %
                        (best_epoch+1, log_dict["val_bag_acc"][best_epoch], log_dict["val_ins_acc"][best_epoch], log_dict["val_mIoU"][best_epoch]))
        if args.is_test == True:
            logging.info('best epoch: %d, test bag acc: %.4f, test inst acc: %.4f, mIoU: %.4f' %
                            (best_epoch+1, log_dict["test_bag_acc"][best_epoch], log_dict["test_ins_acc"][best_epoch], log_dict["test_mIoU"][best_epoch]))

        make_loss_graph(args,log_dict['train_loss'], log_dict['val_loss'], (args.output_path+ "/loss_graph/"+args.mode+"/fold="+ str(args.fold) +"_seed="+str(args.seed)+"_loss_graph.png"))
        make_bag_acc_graph(args, log_dict['train_bag_acc'], log_dict['val_bag_acc'], (args.output_path+ "/acc_graph/"+args.mode+"/fold="+ str(args.fold) +"_seed="+str(args.seed)+"_bag_acc_graph.png"))
        make_ins_acc_graph(args, log_dict['train_ins_acc'], log_dict['val_ins_acc'], (args.output_path+ "/acc_graph/"+args.mode+"/fold="+ str(args.fold) +"_seed="+str(args.seed)+"_ins_acc_graph.png"))
        np.save("%s/log_dict/%s/fold=%d_seed=%d_log" % (args.output_path, args.mode, args.fold, args.seed) , log_dict)
    return