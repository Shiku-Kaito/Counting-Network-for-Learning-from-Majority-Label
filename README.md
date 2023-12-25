# Counting-Network-for-Learning-from-Majority-Label

![Alt Text](./image.jpg)

Shikui Kaito, Shinnosuke Matsuo, Daiki Suehiro, Ryoma Bise
> The paper proposes a novel problem in multi-class Multiple-Instance
Learning (MIL) called Learning from the Majority Label (LML).
In LML, the majority class of instances in a bag is assigned as the
bag’s label. LML aims to classify instances using bag-level majority
classes. This problem is valuable in various applications. Exist
ing MIL methods are unsuitable for LML due to aggregating confi
dences, which may lead to inconsistency between the bag-level label
and the label obtained by counting the number of instances for each
class. This may lead to incorrect instance-level classification. We
propose a counting network trained to produce the bag-level majority
labels estimated by counting the number of instances for each class.
This led to the consistency of the majority class between the net
work outputs and one obtained by counting the number of instances.
Experimental results show that our counting network outperforms
conventional MIL methods on four datasets. Ablation studies fur
ther confirm the counting network superiority.

# Requirement
To set up their environment, please run:  
(we recommend to use [Anaconda](https://www.anaconda.com/) for installation.)
```
conda env create -n LML -f LML.yml
conda activate LML
```
