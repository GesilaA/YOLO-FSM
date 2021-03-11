import os
import random

all_dataset = '../data/train_annotation.txt'
with open(all_dataset, 'rt') as f:
    all_dataset = f.readlines()

with open('../data/all_annotation.txt', 'wt') as f:
    f.writelines(all_dataset)

train_dataset = []
test_dataset = []

for line in all_dataset:
    if random.random() < 0.9:
        train_dataset.append(line)
    else:
        test_dataset.append(line)

with open('../data/train_annotation.txt', 'wt') as f:
    f.writelines(train_dataset)
with open('../data/test_annotation.txt', 'wt') as f:
    f.writelines(test_dataset)