import numpy as np
import scipy.io as sio
from PIL import Image

def load_rgb_img(args, dataset_dir='./dataset/medmnist/'):
    all_data = np.load(dataset_dir + "%smnist.npz" % args.dataset)
    x_train = all_data['train_images']
    y_train = all_data['train_labels']
    x_test = all_data['test_images']
    y_test = all_data['test_labels']
    return x_train, y_train.reshape(-1), x_test, y_test.reshape(-1)

def load_gray_img(args, dataset_dir='./dataset/medmnist/'):
    all_data = np.load(dataset_dir + "%smnist.npz" % args.dataset)
    x_train, y_train, x_test, y_test = all_data['train_images'], all_data['train_labels'], all_data['test_images'], all_data['test_labels']
    train_img_rgb = [np.array(Image.fromarray(img).convert('RGB'))
                        for img in x_train]
    test_img_rgb = [np.array(Image.fromarray(img).convert('RGB'))
                    for img in x_test]
    x_train = np.array(train_img_rgb)
    x_test = np.array(test_img_rgb)
    return x_train, y_train.reshape(-1), x_test, y_test.reshape(-1)

def load_medmnist(args):
    if args.dataset == "blood" or args.dataset == "path":
        x_train, y_train, x_test, y_test = load_rgb_img(args)
    else:
        x_train, y_train, x_test, y_test = load_gray_img(args)

    data = np.concatenate([x_train , x_test], 0)
    label = np.concatenate([y_train , y_test], 0)
    idx = np.arange(len(label))
    np.random.shuffle(idx)
    data , label = data[idx], label[idx]
    return data, label


if __name__ == '__main__':
    train_img, train_label, test_img, test_label = load_medmnist("path")
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
