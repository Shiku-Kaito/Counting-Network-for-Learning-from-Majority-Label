# Counting-Network-for-Learning-from-Majority-Label

![Alt Text](./image.jpg)

Shikui Kaito, Shinnosuke Matsuo, Daiki Suehiro, Ryoma Bise
> The paper proposes a novel problem in multi-class Multiple-Instance Learning (MIL) called Learning from the Majority Label (LML).In LML, the majority class of instances in a bag is assigned as the bag’s label. LML aims to classify instances using bag-level majority classes. This problem is valuable in various applications. Existing MIL methods are unsuitable for LML due to aggregating confidences, which may lead to inconsistency between the bag-level label and the label obtained by counting the number of instances for each class. This may lead to incorrect instance-level classification. We propose a counting network trained to produce the bag-level majority labels estimated by counting the number of instances for each class. This led to the consistency of the majority class between the net
work outputs and one obtained by counting the number of instances. Experimental results show that our counting network outperforms
conventional MIL methods on four datasets. Ablation studies further confirm the counting network superiority.

# Requirement
To set up their environment, please run:  
(we recommend to use [Anaconda](https://www.anaconda.com/) for installation.)
```
conda env create -n LML -f LML.yml
conda activate LML
```

# Dataset
ou can create dataset by running following code. Dataset will be saved in ./data directory.
```
python ./make_bag/crossvali_make_dataset_10class_uniform.py
```

# Training & Test
After creating your python environment and Dataset which can be made by following above command, you can run Counting-Network code.
If you want to train the network, please run following command. 5 fold cross-validation trainning is implemented.
```
python ./main.py --module 'Count' --dataset='cifar10' --classes=10 --majority_size "various" --output_path './result/' --is_evaluation 0
```
If you want to evaluation the network, please run following command. 5 fold cross-validation test is implemented.
```
python ./main.py --module 'Count' --dataset='cifar10' --classes=10 --majority_size "various" --output_path './result/' --is_evaluation 1
```
# Citation
If you find this repository helpful, please consider citing:
```
@INPROCEEDINGS{10448425,
  author={Shiku, Kaito and Matsuo, Shinnosuke and Suehiro, Daiki and Bise, Ryoma},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Counting Network for Learning from Majority Label}, 
  year={2024},
  volume={},
  number={},
  pages={7025-7029},
  keywords={Signal processing;Acoustics;Task analysis;Speech processing;Majority Label;Counting Network;MIL},
  doi={10.1109/ICASSP48485.2024.10448425}}
```

# Author
@ Shiku Kaito  
・ Contact: kaito.shiku@human.ait.kyushu-u.ac.jp
