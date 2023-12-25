import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18
# from additive_modules.transmil import TransMIL
# from additive_modules.VIT import VIT


class Count(nn.Module):
    def __init__(self, num_class, temper1, temper2):
        super().__init__()

        self.feature_extractor = resnet18(pretrained=False)
        self.feature_extractor.fc = nn.Sequential()

        self.temper1 = temper1
        self.temper2 = temper2

        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        (batch, num_ins, c, w, h) = x.size()
        x = x.reshape(-1, c, w, h)

        y = self.feature_extractor(x)
        y_ins = self.classifier(y)
        y_ins = F.softmax(y_ins / self.temper1 ,dim=1)      #softmax with temperature
        count = y_ins.reshape(batch, num_ins, -1)
        count = count.sum(dim=1)      #カウントする
        count /= count.sum(axis=1, keepdims=True)    #正規化
        y_bag = F.softmax(count / self.temper2 ,dim=1)     #softmax with temperature

        return {"bag": y_bag, "prop": count, "ins": y_ins} #,y, y.reshape(-1,num_ins, 512).mean(dim=1)
    

class Count_1d(nn.Module):
    def __init__(self, num_class, temper1, temper2):
        super().__init__()

        self.feature_extractor =  ResNet1D(
        in_channels=13, 
        base_filters=128, # 64 for ResNet1D, 352 for ResNeXt1D
        kernel_size=16, 
        stride=2, 
        groups=32, 
        n_block=24, 
        n_classes=None, 
        downsample_gap=6, 
        increasefilter_gap=12, 
        use_do=True)

        self.temper1 = temper1
        self.temper2 = temper2

        self.classifier = nn.Linear(256, num_class)

    def forward(self, x):
        (batch, num_ins, c, l) = x.size()
        x = x.reshape(-1, c, l)

        y = self.feature_extractor(x)
        y_ins = self.classifier(y)
        y_ins = F.softmax(y_ins / self.temper1 ,dim=1)      #softmax with temperature

        count = y_ins.reshape(batch, num_ins, -1)
        count = count.sum(dim=1)      #カウントする
        count /= count.sum(axis=1, keepdims=True)    #正規化
        y_bag = F.softmax(count / self.temper2 ,dim=1)     #softmax with temperature

        return {"bag": y_bag, "ins": y_ins} #,y, y.reshape(-1,num_ins, 512).mean(dim=1)
    

class Count_ent(nn.Module):
    def __init__(self, num_class, temper1, temper2):
        super().__init__()

        self.feature_extractor = resnet18(pretrained=False)
        self.feature_extractor.fc = nn.Sequential()

        self.temper1 = temper1
        self.temper2 = temper2

        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        (batch, num_ins, c, w, h) = x.size()
        x = x.reshape(-1, c, w, h)

        y = self.feature_extractor(x)
        y_out = self.classifier(y)
        # y_val = y_out / y_out.sum(axis=1, keepdims=True)    #正規化
        # y_val = torch.sigmoid(y_out)
        y_ins = F.softmax(y_out / self.temper1 ,dim=1)      #softmax with temperature

        count = y_ins.reshape(batch, num_ins, -1)
        count = count.sum(dim=1)      #カウントする
        count /= count.sum(axis=1, keepdims=True)    #正規化
        y_bag = F.softmax(count / self.temper2 ,dim=1)     #softmax with temperature

        y_out = y_out.reshape(batch, num_ins, -1)
        return y_bag, y_ins , y_out
    
class Count_ent_expval(nn.Module):
    def __init__(self, num_class, temper1, temper2):
        super().__init__()

        self.feature_extractor = resnet18(pretrained=False)
        self.feature_extractor.fc = nn.Sequential()

        self.temper1 = temper1
        self.temper2 = temper2

        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        (batch, num_ins, c, w, h) = x.size()
        x = x.reshape(-1, c, w, h)

        y = self.feature_extractor(x)
        y_out = self.classifier(y)
        # y_val = y_out / y_out.sum(axis=1, keepdims=True)    #正規化
        y_ins = F.softmax(y_out / self.temper1 ,dim=1)      #softmax with temperature

        count = y_ins.reshape(batch, num_ins, -1)
        count = count.sum(dim=1)      #カウントする
        count /= count.sum(axis=1, keepdims=True)    #正規化
        y_bag = F.softmax(count / self.temper2 ,dim=1)     #softmax with temperature

        y_conf = F.softmax(y_out ,dim=1)      #softmax
        y_expval = y_conf.reshape(batch, num_ins, -1)
        y_expval = y_expval.sum(dim=1)      #カウントする
        y_expval /= y_expval.sum(axis=1, keepdims=True)    #正規化
        y_expval = F.softmax(y_expval / self.temper2 ,dim=1)     #softmax with temperature

        y_out = y_out.reshape(batch, num_ins, -1)
        return y_bag, y_ins , y_out, y_expval 
    

# class Count_multitask(nn.Module):
#     def __init__(self, args):
#         super().__init__()

#         self.feature_extractor = resnet18(pretrained=False)
#         self.feature_extractor.fc = nn.Sequential()

#         self.temper1 = args.temper1
#         self.temper2 = args.temper2

#         self.ins_classifier = nn.Linear(512, args.classes)
#         self.bag_classifier = nn.Linear(512, args.classes)

#         self.multitask_mode = args.multitask_mode
#         if self.multitask_mode == "transmil":
#             self.multisk_model = TransMIL(args.classes)
#         if self.multitask_mode == "VIT":
#             self.multisk_model = VIT(args.classes, device = args.device)

#     def forward(self, x):
#         (batch, num_ins, c, w, h) = x.size()
#         x = x.reshape(-1, c, w, h)

#         #count
#         y = self.feature_extractor(x)
#         y_out = self.ins_classifier(y)
#         # y_val = y_out / y_out.sum(axis=1, keepdims=True)    #正規化
#         y_ins = F.softmax(y_out / self.temper1 ,dim=1)      #softmax with temperature

#         count = y_ins.reshape(batch, num_ins, -1)
#         count = count.sum(dim=1)      #カウントする
#         count /= count.sum(axis=1, keepdims=True)    #正規化
#         y_bag = F.softmax(count / self.temper2 ,dim=1)     #softmax with temperature

#         #bag feature
#         if self.multitask_mode == "transmil" or self.multitask_mode == "VIT":
#             y_feature = y.reshape(batch, num_ins, -1)
#             y_bag_feat_dict = self.multisk_model(y_feature)
#             y_bag_feat = y_bag_feat_dict['Y_prob']

#         else:
#             y_feature = y.reshape(batch, num_ins, -1)
#             mean_feature = y_feature.mean(dim=1)      #特徴を平均
#             y_bag_feat = self.bag_classifier(mean_feature) 
#             y_bag_feat = F.softmax(y_bag_feat ,dim=1) 

#         y_out = y_out.reshape(batch, num_ins, -1)
#         return y_bag, y_ins , y_out, y_bag_feat ,y_feature.reshape(-1, 512), mean_feature
    

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = False
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        # self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        # if self.verbose:
        #     print('final pooling', out.shape)
        # # out = self.do(out)
        # out = self.dense(out)
        # if self.verbose:
        #     print('dense', out.shape)
        # # out = self.softmax(out)
        # if self.verbose:
        #     print('softmax', out.shape)
        
        return out    