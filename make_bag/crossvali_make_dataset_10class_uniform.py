import argparse
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import make_dataset_folder
from load_svhn import load_svhn
from load_cifar10 import load_cifar10
from load_medmnist import load_medmnist
from load_SpokenArabicDigits import load_SpokenArabicDigits
from glob import glob
from PIL import Image
from tqdm import tqdm


def shuffle_along_axis(proportion, axis):
    idx = np.random.rand(*proportion.shape).argsort(axis=axis)
    return np.take_along_axis(proportion,idx,axis=axis)

def get_label_proportion_large(num_bags, num_classes, args):    #majoruty proportion: uniform distribution 0.6 to 1.0
    split_num = 10
    proportion = [[] for i in range(split_num)]
    #two uniform distribution for making 0.6 to 0.9 majority proportion
    cnt = np.array([0] * split_num)
    while True:
        for i in  range(split_num):
            if len(proportion[i])==(num_bags/split_num):
                cnt[i]=1
        if sum(cnt[:4]==1) == 4:
            break
        tmp_prop_mino = np.random.uniform(low=0.0, high=0.3, size=(10000000, num_classes-1))
        tmp_prop_major = np.random.uniform(low=0.8, high=1, size=(10000000, 1))
        tmp_prop = np.concatenate([tmp_prop_mino, tmp_prop_major], 1)
        tmp_prop /= tmp_prop.sum(axis=1, keepdims=True)
        for i in range(split_num):
            major_prop = np.max(tmp_prop, axis=1)
            partical_prop = tmp_prop[(major_prop>(0.6 + ((1-0.6)/split_num)*i)) * (major_prop<(0.6 + ((1-0.6)/split_num)*(i+1)))]
            proportion[i].extend(partical_prop)
            proportion[i] = proportion[i][:int(num_bags/split_num)]

     #two uniform distribution for making 0.9 to 1 majority proportion
    while True:
        for i in  range(split_num):
            if len(proportion[i])==(num_bags/split_num):
                cnt[i]=1
        if sum(cnt==1) >= 7:
            break
        tmp_prop_mino = np.random.uniform(low=0.0, high=0.1, size=(10000000, num_classes-1))
        tmp_prop_major = np.random.uniform(low=0.9, high=1, size=(10000000, 1))
        tmp_prop = np.concatenate([tmp_prop_mino, tmp_prop_major], 1)
        tmp_prop /= tmp_prop.sum(axis=1, keepdims=True)
        for i in range(split_num):
            major_prop = np.max(tmp_prop, axis=1)
            partical_prop = tmp_prop[(major_prop>(0.6 + ((1-0.6)/split_num)*i)) * (major_prop<(0.6 + ((1-0.6)/split_num)*(i+1)))]
            proportion[i].extend(partical_prop)
            proportion[i] = proportion[i][:int(num_bags/split_num)]

    while True:
        for i in  range(split_num):
            if len(proportion[i])==(num_bags/split_num):
                cnt[i]=1
        if sum(cnt==1) == split_num:
            break
        tmp_prop_mino = np.random.uniform(low=0.0, high=0.05, size=(10000000, num_classes-1))
        tmp_prop_major = np.random.uniform(low=0.95, high=1, size=(10000000, 1))
        tmp_prop = np.concatenate([tmp_prop_mino, tmp_prop_major], 1)
        tmp_prop /= tmp_prop.sum(axis=1, keepdims=True)
        for i in range(split_num):
            major_prop = np.max(tmp_prop, axis=1)
            partical_prop = tmp_prop[(major_prop>(0.6 + ((1-0.6)/split_num)*i)) * (major_prop<(0.6 + ((1-0.6)/split_num)*(i+1)))]
            proportion[i].extend(partical_prop)
            proportion[i] = proportion[i][:int(num_bags/split_num)]

    proportion=np.array(proportion)
    proportion=np.reshape(proportion, (num_bags, num_classes))
    np.random.shuffle(proportion)
    return proportion

def get_label_proportion_small(num_bags, num_classes, args):   #majoruty proportion: uniform distribution (1/classnum) to 0.4
    split_num = 10
    proportion = [[] for i in range(split_num)]
    #one uniform distribution for making 1/classnum to 0.7 majority proportion
    while True:
        cnt = np.array([0] * split_num)
        for i in  range(split_num):
            if len(proportion[i])==(num_bags/split_num):
                cnt[i]=1
        if sum(cnt==1) == 10:
            break
        tmp_prop = np.random.uniform(low=0.0, high=1.0, size=(10000000, num_classes))
        tmp_prop /= tmp_prop.sum(axis=1, keepdims=True)
        for i in range(split_num):
            major_prop = np.max(tmp_prop, axis=1)
            partical_prop = tmp_prop[(major_prop>(1/num_classes + ((0.4-1/num_classes)/split_num)*i)) * (major_prop<(1/num_classes + ((0.4-1/num_classes)/split_num)*(i+1)))]
            proportion[i].extend(partical_prop)
            proportion[i] = proportion[i][:int(num_bags/split_num)]


    proportion=np.array(proportion)
    proportion=np.reshape(proportion, (num_bags, num_classes))
    np.random.shuffle(proportion)
    return proportion

def get_label_proportion_various(num_bags, num_classes, args):  #majoruty proportion: uniform distribution (1/classnum) to 0.4
    split_num = 10
    proportion = [[] for i in range(split_num)]
    #one uniform distribution for making 1/classnum to 0.7 majority proportion
    while True:
        cnt = np.array([0] * split_num)
        for i in  range(split_num):
            if len(proportion[i])==int(num_bags/split_num):
                cnt[i]=1
        if sum(cnt[:5]==1) == 5:
            break
        tmp_prop = np.random.uniform(low=0.0, high=1.0, size=(10000000, num_classes))
        tmp_prop /= tmp_prop.sum(axis=1, keepdims=True)
        for i in range(split_num):
            major_prop = np.max(tmp_prop, axis=1)
            partical_prop = tmp_prop[(major_prop>(1/num_classes + ((1-1/num_classes)/split_num)*i)) * (major_prop<(1/num_classes + ((1-1/num_classes)/split_num)*(i+1)))]
            proportion[i].extend(partical_prop)
            proportion[i] = proportion[i][:int(num_bags/split_num)]

    #two uniform distribution for making 0.7 to 0.9 majority proportion
    while True:
        cnt = np.array([0] * split_num)
        for i in  range(split_num):
            if len(proportion[i])==int(num_bags/split_num):
                cnt[i]=1
        if sum(cnt[:8]==1) == 8:
            break
        tmp_prop_mino = np.random.uniform(low=0.0, high=0.3, size=(10000000, num_classes-1))
        tmp_prop_major = np.random.uniform(low=0.8, high=1, size=(10000000, 1))
        tmp_prop = np.concatenate([tmp_prop_mino, tmp_prop_major], 1)
        tmp_prop /= tmp_prop.sum(axis=1, keepdims=True)
        for i in range(split_num):
            major_prop = np.max(tmp_prop, axis=1)
            partical_prop = tmp_prop[(major_prop>(1/num_classes + ((1-1/num_classes)/split_num)*i)) * (major_prop<(1/num_classes + ((1-1/num_classes)/split_num)*(i+1)))]
            proportion[i].extend(partical_prop)
            proportion[i] = proportion[i][:int(num_bags/split_num)]

    #two uniform distribution for making 0.9 to 1s majority proportion
    while True:
        cnt = np.array([0] * split_num)
        for i in  range(split_num):
            if len(proportion[i])==int(num_bags/split_num):
                cnt[i]=1
        if sum(cnt==1) == 10:
            break
        tmp_prop_mino = np.random.uniform(low=0.0, high=0.1, size=(10000000, num_classes-1))
        tmp_prop_major = np.random.uniform(low=0.9, high=1, size=(10000000, 1))
        tmp_prop = np.concatenate([tmp_prop_mino, tmp_prop_major], 1)
        tmp_prop /= tmp_prop.sum(axis=1, keepdims=True)
        for i in range(split_num):
            major_prop = np.max(tmp_prop, axis=1)
            partical_prop = tmp_prop[(major_prop>(1/num_classes + ((1-1/num_classes)/split_num)*i)) * (major_prop<(1/num_classes + ((1-1/num_classes)/split_num)*(i+1)))]
            proportion[i].extend(partical_prop)
            proportion[i] = proportion[i][:int(num_bags/split_num)]
    bag_chek=0
    for i in range(split_num):
        bag_chek += len(proportion[i])
    if bag_chek < num_bags:
        tmp=[]
        tmp_prop_mino = np.random.uniform(low=0.0, high=0.3, size=(10000000, num_classes-1))
        tmp_prop_major = np.random.uniform(low=0.8, high=1, size=(10000000, 1))
        tmp_prop = np.concatenate([tmp_prop_mino, tmp_prop_major], 1)
        tmp_prop /= tmp_prop.sum(axis=1, keepdims=True)
        major_prop = np.max(tmp_prop, axis=1)
        partical_prop = tmp_prop[(major_prop>(1/num_classes))]
        for i in range(split_num):
            for a in proportion[i]:
                tmp.append(a)
        for a in partical_prop[:(num_bags-bag_chek)]:
            tmp.append(a)
        proportion = np.array(tmp)

    proportion=np.array(proportion)
    proportion=np.reshape(proportion, (num_bags, num_classes))
    np.random.shuffle(proportion)
    return proportion


def get_N_label_proportion(proportion, num_instances, num_classes):
    N = np.zeros(proportion.shape)
    for i in range(len(proportion)):
        p = proportion[i]
        for c in range(len(p)):
            if (c+1) != num_classes:
                num_c = int(np.round(num_instances*p[c]))     
                if sum(N[i])+num_c >= num_instances:
                    num_c = int(num_instances-sum(N[i]))      
                    print("++++++++++++++++over++++++++++++++")
                    print(c)
            else:

                num_c = int(num_instances-sum(N[i]))         

            N[i][c] = int(num_c)
        np.random.shuffle(N[i])                              
    print(N.sum(axis=0))                                     
    print((N.sum(axis=1) != num_instances).sum())             
    return N


def create_bags(data, label, num_bags, args):
    # make poroportion
    if args.majority_size == "small":
        proportion = get_label_proportion_small(num_bags, args.num_classes, args)       #dicide the proportion 
    elif args.majority_size == "various":
        proportion = get_label_proportion_various(num_bags, args.num_classes, args)     #dicide the proportion 
    elif args.majority_size == "large":
        proportion = get_label_proportion_large(num_bags, args.num_classes, args)        #dicide the proportion  
    proportion_N = get_N_label_proportion(                                      #obtain the instance data correspond to the proportion
        proportion, args.num_instances, args.num_classes)

    # make index
    idx = np.arange(len(label))
    idx_c = []
    for c in range(args.num_classes): 
        idx_c.append(idx[label[idx] == c])
    for i in range(len(idx_c)):
        random.shuffle(idx_c[i])


    bags_idx = []
    if args.overlap == 1:   #instance overlap
        for n in range(len(proportion_N)):   
            bag_idx = []
            for c in range(args.num_classes):
                sample_c_index = np.random.choice(idx_c[c], size=int(proportion_N[n][c]), replace=False)   #overlap
                bag_idx.extend(sample_c_index)                                                             
            np.random.shuffle(bag_idx)
            if len(bag_idx) < args.num_instances:
                break
            bags_idx.append(bag_idx)  

    else:    #instance non overlap
        for n in range(len(proportion_N)):    
            bag_idx = []
            for c in range(args.num_classes):
                sample_c_index = idx_c[c][0:int(proportion_N[n][c])]   # non overlap
                idx_c[c] = idx_c[c][int(proportion_N[n][c]):]   # non overlap
                bag_idx.extend(sample_c_index)                              
            np.random.shuffle(bag_idx)
            if len(bag_idx) < args.num_instances:
                break
            bags_idx.append(bag_idx)                                                            

    bags_idx=np.array(bags_idx)

    # make data, label, proportion
    bags, labels = data[bags_idx], label[bags_idx]     
    original_lps = proportion_N / args.num_instances
    major_labels=np.zeros(original_lps.shape)
    major_labels=np.argmax(original_lps,axis=1)

    return bags, labels, major_labels 

def main(args):
    # load dataset
    if args.dataset == 'svhn':
        data, label = load_svhn()
    elif args.dataset == 'cifar10':
        data, label = load_cifar10()
    elif args.dataset == 'blood' or args.dataset == 'oct' or args.dataset == 'organa' or args.dataset == 'organc' or args.dataset == 'organs' or args.dataset == 'path':
        data, label = load_medmnist(args)
    elif args.dataset == 'SpokenArabicDigits':
        data, label = load_SpokenArabicDigits()


    # k-fold cross validation
    split_len = int(len(data)/5)
    fold_dict = {'data0': data[0:split_len] ,'label0': label[0:split_len], 'data1': data[split_len:(split_len*2)] ,'label1': label[split_len:(split_len*2)],  'data2':  data[(split_len*2):(split_len*3)], 'label2': label[(split_len*2):(split_len*3)], 'data3': data[(split_len*3):(split_len*4)], 'label3': label[(split_len*3):(split_len*4)], 'data4': data[(split_len*4):(split_len*5)], 'label4': label[(split_len*4):(split_len*5)]}

    for i in range(5):
        test_data, test_label = fold_dict['data%d'%(i%5)], fold_dict['label%d'%(i%5)]
        val_data, val_label = fold_dict['data%d'%((1+i)%5)], fold_dict['label%d'%((1+i)%5)]
        train_data, train_label = np.concatenate([fold_dict['data%d'%((2+i)%5)], fold_dict['data%d'%((3+i)%5)], fold_dict['data%d'%((4+i)%5)]])  ,np.concatenate([fold_dict['label%d'%((2+i)%5)], fold_dict['label%d'%((3+i)%5)], fold_dict['label%d'%((4+i)%5)]])               #data内のインデックス指定して,train,validationに分ける

        output_path = './data/%s/%dclass_%s/%d/' % (
        args.dataset, args.num_classes, args.majority_size, i)

        make_dataset_folder(output_path)

        f = open((output_path + 'bag_info.txt'), 'w')
        f.write('$ shape=(bag num , bag size, hight, width, channel) \n')

        # train
        bags, labels, major_labels = create_bags(train_data, train_label,
                                                            args.train_num_bags,args)                                                  
        np.save('%s/train_bags' % (output_path), bags)
        np.save('%s/train_ins_labels' % (output_path), labels)
        np.save('%s/train_major_labels' % (output_path), major_labels)
        f.write('$ training \n data shape=' + str(bags.shape) + '\n')


        bags, labels, major_labels = create_bags(val_data, val_label,
                                                            args.val_num_bags,
                                                            args) 
        np.save('%s/val_bags' % (output_path), bags)
        np.save('%s/val_ins_labels' % (output_path), labels)
        np.save('%s/val_major_labels' % (output_path), major_labels)
        f.write('$ validation \n data shape=' + str(bags.shape) + '\n')

        # test
        bags, labels, major_labels = create_bags(test_data, test_label,
                                                                args.test_num_bags,
                                                                args)  
        np.save('%s/test_bags' % (output_path), bags)
        np.save('%s/test_ins_labels' % (output_path), labels)
        np.save('%s/test_major_labels' % (output_path), major_labels)
        f.write('$ test \n data shape=' + str(bags.shape) + '\n')

        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--dataset', default='svhn', 
                        type=str, help='cifar10 or mnist or svhn or blood or oct or organa or organc or organs or path or NCT')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_instances', default=64, type=int)

    parser.add_argument('--train_num_bags', default=400, type=int)
    parser.add_argument('--val_num_bags', default=100, type=int)
    parser.add_argument('--test_num_bags', default=100, type=int)

    parser.add_argument('--majority_size', default='various', type=str, help='small or various or large')
    parser.add_argument('--bag_size', default='bagsize=32_small', type=str, help='bagsize=16 or bagsize=256')
    parser.add_argument('--overlap', default=1, type=int, help='1 or 0')
    args = parser.parse_args()

    #################
    np.random.seed(args.seed)
    main(args)

