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
from load_mnist import load_mnist
from load_medmnist import load_medmnist
from load_NCT import load_NCT
from glob import glob
from PIL import Image
from tqdm import tqdm

def shuffle_along_axis(proportion, axis):
    idx = np.random.rand(*proportion.shape).argsort(axis=axis)
    return np.take_along_axis(proportion,idx,axis=axis)

def get_label_proportion_large(num_bags, num_classes, args):   #majoruty proportionが0.6~1の間でユニフォーム
    split_num = 10
    proportion = [[] for i in range(split_num)]
    #2つの一様分布を使って、0.7<major_prop<0.9くらいのproportionを作成
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

    #2つの一様分布を使って、0.9<major_prop<くらいのproportionを作成
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

def get_label_proportion_small(num_bags, num_classes, args):   #majoruty proportionが (1/クラス数) ~ 0.4の間でユニフォーム
    split_num = 10
    proportion = [[] for i in range(split_num)]
    #1つの一様分布を使って、0.7くらいまでのproportionを作成
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

def get_label_proportion(num_bags, num_classes, args):
    split_num = 10
    proportion = [[] for i in range(split_num)]
    #1つの一様分布を使って、0.7くらいまでのproportionを作成
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

    #2つの一様分布を使って、0.7<major_prop<0.9くらいのproportionを作成
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

    #2つの一様分布を使って、0.9<major_prop<くらいのproportionを作成
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
                num_c = int(np.round(num_instances*p[c]))     #バッグ全体のインスタンス数にproportionをかけて、クラスのインスタンス数を決める
                if sum(N[i])+num_c >= num_instances:
                    num_c = int(num_instances-sum(N[i]))      #インスタンス数が、バッグ内のインスタンス数を超えてないかチェック
                    print("++++++++++++++++over++++++++++++++")
                    print(c)
            else:
                # if int(num_instances-sum(N[i])) == 0:
                    # print("++++++++++++++++over++++++++++++++")
                num_c = int(num_instances-sum(N[i]))          #最後のクラスは、バックの空いてる分の数

            N[i][c] = int(num_c)
        np.random.shuffle(N[i])                               #バック内のクラスのインスタンス数をシャッフル
    print(N.sum(axis=0))                                      #全体のクラスのインスタンス数にかたよりないかチェック
    print((N.sum(axis=1) != num_instances).sum())             #設定したバッグ内のインスタンス数を超えてるのないかチェック
    return N


def create_bags(data, label, num_bags, args):
    # make poroportion
    if args.majority_size == "small":
        proportion = get_label_proportion_small(num_bags, args.num_classes, args)
    elif args.majority_size == "usual":
        proportion = get_label_proportion(num_bags, args.num_classes, args)          #各バッグ内のプロポーションを決める
    elif args.majority_size == "large":
        proportion = get_label_proportion_large(num_bags, args.num_classes, args)          #各バッグ内のプロポーションを決める    
    proportion_N = get_N_label_proportion(                                      #決めたプロポーションに合わせて、各のクラスのバッグ内のインスタンス数を決める
        proportion, args.num_instances, args.num_classes)

    # proportion_N_nega = np.zeros((num_nega_bags, args.num_classes))
    # proportion_N_nega[:, 0] = args.num_instances

    # proportion_N = np.concatenate([proportion_N, proportion_N_nega], axis=0)

    # make index
    idx = np.arange(len(label))
    idx_c = []
    for c in range(args.num_classes):   #10クラス分あるデータ内から、使う3クラス分取り出す
        idx_c.append(idx[label[idx] == c])
    for i in range(len(idx_c)):
        random.shuffle(idx_c[i])

    # bags_idx = []
    # num_used_idx = np.zeros(args.num_classes).astype(int)
    # for n in range(len(proportion_N)):
    #     bag_idx = []
    #     for c in range(args.num_classes):
    #         N = int(proportion_N[n][c])
    #         sample_c_index = idx_c[c][num_used_idx[c]: num_used_idx[c]+N]
    #         bag_idx.extend(sample_c_index)
    #         num_used_idx[c] += N

    #     np.random.shuffle(bag_idx)
    #     bags_idx.append(bag_idx)

    bags_idx = []
    if args.overlap == 1:    #重複あり
        for n in range(len(proportion_N)):    #len(proportion_N)はバッグ数
            bag_idx = []
            for c in range(args.num_classes):
                # sample_c_index = np.random.choice(idx_c[c], size=int(proportion_N[n][c]), replace=False)    #3つ分のクラスのインデックスの中から、指定したクラスのインデックスを、proportionで指定された数読み込む
                sample_c_index = np.random.choice(idx_c[c], size=int(proportion_N[n][c]), replace=False)   #重複あり
                bag_idx.extend(sample_c_index)                                                              #bag_idxにデータじゃなく、取りたいデータのインデックスを入れる
            np.random.shuffle(bag_idx)
            if len(bag_idx) < args.num_instances:
                break
            bags_idx.append(bag_idx)               #全部のバッグ分   
    else:       #重複なし
        for n in range(len(proportion_N)):    #len(proportion_N)はバッグ数
            bag_idx = []
            for c in range(args.num_classes):
                # sample_c_index = np.random.choice(idx_c[c], size=int(proportion_N[n][c]), replace=False)    #3つ分のクラスのインデックスの中から、指定したクラスのインデックスを、proportionで指定された数読み込む
                sample_c_index = idx_c[c][0:int(proportion_N[n][c])]   #重複なしver
                idx_c[c] = idx_c[c][int(proportion_N[n][c]):]   #重複なしver
                bag_idx.extend(sample_c_index)                                                              #bag_idxにデータじゃなく、取りたいデータのインデックスを入れる
            np.random.shuffle(bag_idx)
            if len(bag_idx) < args.num_instances:
                break
            bags_idx.append(bag_idx)               #全部のバッグ分                                                                
    # bags_index.shape => (num_bags, num_instances)

    bags_idx=np.array(bags_idx)

    # make data, label, proportion
    bags, labels = data[bags_idx], label[bags_idx]        #インデックスをもとに各バッグにデータを入れる
    original_lps = proportion_N / args.num_instances      #松尾さんの場合はproportinがラベルになるから、保存
    major_labels=np.zeros(original_lps.shape)
    major_labels=np.argmax(original_lps,axis=1)
    # for idx, major_idx in enumerate(major_idies):
    #     major_labels[idx][major_idx]=1
        
    # partial_lps = original_lps.copy()
    # posi_nega = (original_lps[:, 0] != 1)
    # partial_lps[posi_nega == 1, 0] = 0  # mask negative class
    # partial_lps /= partial_lps.sum(axis=1, keepdims=True)  # normalize

    return bags, labels, major_labels 

def main(args):
    # load dataset
    data, label = load_medmnist(args)


    # k-fold cross validation
    split_len = int(len(data)/5)
    fold_dict = {'data0': data[0:split_len] ,'label0': label[0:split_len], 'data1': data[split_len:(split_len*2)] ,'label1': label[split_len:(split_len*2)],  'data2':  data[(split_len*2):(split_len*3)], 'label2': label[(split_len*2):(split_len*3)], 'data3': data[(split_len*3):(split_len*4)], 'label3': label[(split_len*3):(split_len*4)], 'data4': data[(split_len*4):(split_len*5)], 'label4': label[(split_len*4):(split_len*5)]}

    for i in range(5):
        test_data, test_label = fold_dict['data%d'%(i%5)], fold_dict['label%d'%(i%5)]
        val_data, val_label = fold_dict['data%d'%((1+i)%5)], fold_dict['label%d'%((1+i)%5)]
        train_data, train_label = np.concatenate([fold_dict['data%d'%((2+i)%5)], fold_dict['data%d'%((3+i)%5)], fold_dict['data%d'%((4+i)%5)]])  ,np.concatenate([fold_dict['label%d'%((2+i)%5)], fold_dict['label%d'%((3+i)%5)], fold_dict['label%d'%((4+i)%5)]])               #data内のインデックス指定して,train,validationに分ける

        if args.majority_size == 'small' or args.majority_size == 'large':
            output_path = './data/%s/%dclass_more_majority_uniform_%s/%d/' % (
            args.dataset, args.num_classes, args.majority_size, i)
        elif args.majority_size == 'usual':
            output_path = './data/%s/%dclass_more_majority_uniform_%s/%d/' % (
            args.dataset, args.num_classes, args.bag_size, i)
        make_dataset_folder(output_path)

        f = open((output_path + 'bag_info.txt'), 'w')
        f.write('$ shape=(bag num , bag size, hight, width, channel) \n')

        # train
        while(1):
            bags, labels, major_labels = create_bags(train_data, train_label,
                                                                args.train_num_bags,args)
            if len(bags)==args.train_num_bags:
                break                                                    
        np.save('%s/train_bags' % (output_path), bags)
        np.save('%s/train_ins_labels' % (output_path), labels)
        np.save('%s/train_major_labels' % (output_path), major_labels)
        f.write('$ training \n data shape=' + str(bags.shape) + '\n')

        # val
        while(1):
            bags, labels, major_labels = create_bags(val_data, val_label,
                                                                args.val_num_bags,
                                                                args)
            if len(bags)==args.val_num_bags:
                break       
        np.save('%s/val_bags' % (output_path), bags)
        np.save('%s/val_ins_labels' % (output_path), labels)
        np.save('%s/val_major_labels' % (output_path), major_labels)
        f.write('$ validation \n data shape=' + str(bags.shape) + '\n')

        # test

        # used_test_data, used_test_label = [], []
        # for c in range(args.num_classes):
        #     used_test_data.extend(test_data[test_label == c])
        #     used_test_label.extend(test_label[test_label == c])
        # test_data, test_label = np.array(used_test_data), np.array(used_test_label)

        while(1):
            bags, labels, major_labels = create_bags(test_data, test_label,
                                                                    args.test_num_bags,
                                                                    args)
            if len(bags)==args.test_num_bags:
                break   
        np.save('%s/test_bags' % (output_path), bags)
        np.save('%s/test_ins_labels' % (output_path), labels)
        np.save('%s/test_major_labels' % (output_path), major_labels)
        f.write('$ test \n data shape=' + str(bags.shape) + '\n')

        f.close()

    # np.save('data/%s/%dclass/test_data' %
    #         (args.dataset, args.num_classes), test_data)
    # np.save('data/%s/%dclass/test_label' %
    #         (args.dataset, args.num_classes), test_label)


# def CRC100K():
#     negative_class = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM']
#     positive_class = ['STR', 'TUM']

#     # negative_path_list, positive_path_list = [], []
#     data, label = [], []
#     for c in negative_class:
#         for p in tqdm(glob('../dataset/NCT-CRC-HE-100K/%s/*' % c)):
#             data.append(np.asarray(Image.open(p).convert('RGB')))
#             label.append(0)

#     for i, c in tqdm(enumerate(positive_class)):
#         for p in glob('../dataset/NCT-CRC-HE-100K/%s/*' % c):
#             data.append(np.asarray(Image.open(p).convert('RGB')))
#             label.append(i+1)

#     np.save('../dataset/NCT-CRC-HE-100K/data', np.array(data))
#     np.save('../dataset/NCT-CRC-HE-100K/label', np.array(label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--dataset', default='cifar10', 
                        type=str, help='cifar10 or mnist or svhn or blood or oct or organa or organc or organs or path or NCT')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_instances', default=256, type=int)

    parser.add_argument('--train_num_bags', default=100, type=int)
    parser.add_argument('--val_num_bags', default=25, type=int)
    parser.add_argument('--test_num_bags', default=25, type=int)

    # parser.add_argument('--minor_mean', default=0.33, type=float)
    # parser.add_argument('--major_mean', default=0.7, type=float)
    # parser.add_argument('--minor_despersion', default=0.3, type=float)
    # parser.add_argument('--major_despersion', default=0.3, type=float)
    parser.add_argument('--majority_size', default='usual', type=str, help='small or usual or large')
    parser.add_argument('--bag_size', default='bagsize=256_bagnum=100', type=str, help='bagsize=16 or bagsize=256')
    parser.add_argument('--overlap', default=0, type=int, help='1 or 0')
    args = parser.parse_args()

    #################
    np.random.seed(args.seed)
    main(args)
    # CRC100K()
