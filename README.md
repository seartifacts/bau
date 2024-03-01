# BAMU
The code for paper 'Backdoor Attack through Machine Unlearning' which was submitted to TIFS, is available here.

# Environment
This project is based on python 3.8 and pytorch 1.12.

# Structure
Input-Targeted-based BAMU: bamu/attack/it<br>
BadNets-based BAMU: bamu/attack/bn<br>
Neural Cleanse: bamu/attack/nc<br>
Randomized Channel Shuffling: bamu/attack/rcs<br>
Model-Uncertainty-based Detection: bamu/defense/mu<br>
Sub-Model-Similarity-based Detection: bamu/defense/sms

# Usage
Attack:
```
python3 cifar10_bn.py --experiment=0 --gpu=0 --path=modelpath --epochs=100 --shards=1 --slices=1 --poison_num=300 --mitigation_num=100 --requests=0
```
Detection:
```
python3 fmnist_it_mu.py --experiment=0 --gpu=0 --path=modelpath --shards=1 --slices=1 --poison_num=5 --mitigation_num=15 --requests=0
```
