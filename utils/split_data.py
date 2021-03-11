# 将数据集分离成大、中、小三种尺度
import random

txt_file = '../data/train_annotation.txt'
fs = '../data/train_annotation_small.txt'
fm = '../data/train_annotation_medium.txt'
fl = '../data/train_annotation_large.txt'

small_dataset = []
medium_dataset = []
large_dataset = []

f = open(txt_file, 'rt')
all_dataset = f.readlines()
f.close()

for line in all_dataset:
    items = line.split(' ')[1:]
    small_tag = False
    medium_tag = False
    large_tag = False
    for obj in items:
        x1, y1, x2, y2, cls = obj.split(',')
        scale = (int(x2)-int(x1)) * (int(y2)-int(y1))
        if scale < 32 * 32:
            small_tag = True
        elif scale < 96 * 96:
            medium_tag = True
        else:
            large_tag = True
    if small_tag:
        small_dataset.append(line)
    if medium_tag:
        medium_dataset.append(line)
    if large_tag:
        large_dataset.append(line)

small_dataset = small_dataset * 12
medium_dataset = medium_dataset * 3
random.shuffle(small_dataset)
random.shuffle(medium_dataset)
random.shuffle(large_dataset)
fs = open(fs, 'wt')
fs.writelines(small_dataset)
fs.close()
fm = open(fm, 'wt')
fm.writelines(medium_dataset)
fm.close()
fl = open(fl, 'wt')
fl.writelines(large_dataset)
fl.close()
