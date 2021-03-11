import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

CLASS_MAP = {
    'aeroplane'     : (128, 0  , 0  ),
    'bicycle'       : (0  , 128, 0  ),
    'bird'          : (128, 128, 0  ),
    'boat'          : (0  , 0  , 128),
    'bottle'        : (128, 0  , 128),
    'bus'           : (0  , 128, 128),
    'car'           : (128, 128, 128),
    'cat'           : (64 , 0  , 0  ),
    'chair'         : (192, 0  , 0  ),
    'cow'           : (64 , 128, 0  ),
    'diningtable'   : (192, 128, 0  ),
    'dog'           : (64 , 0  , 128),
    'horse'         : (192, 0  , 128),
    'motorbike'     : (64 , 128, 128),
    'person'        : (192, 128, 128),
    'pottedplant'   : (0  , 64 , 0  ),
    'sheep'         : (128, 64 , 0  ),
    'sofa'          : (0  , 192, 0  ),
    'train'         : (128, 192, 0  ),
    'monitor'       : (0  , 64 , 128),
}

COLOR_MAP = {
    v : k for k, v in CLASS_MAP.items()
}

BORDER = {
    'border': (224, 224, 192),
}

def border_delete(dataset):
    aot_dir, _ = dataset
    new_dir = os.path.join(aot_dir, '../SegmentationClassWithoutBorder')
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for mask_path in glob.glob(os.path.join(aot_dir, '*.png')):
        img_mask = cv2.imread(mask_path)
        rgb = BORDER['border']
        bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)
        img_mask[img_mask == bgr] = 0
        cv2.imwrite(os.path.join(new_dir, os.path.split(mask_path)[1]), img_mask)

def find_obj_pos(img):
    obj_pos = []
    background = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    while True:
        h, w = np.where(background > 0)
        if len(h) <= 0:
            break
        h, w = h[0], w[0]
        obj_pos.append((h, w))
        background[background == background[h, w]] = 0
    return obj_pos

def unpadding(obj_img):
    obj_gray = cv2.cvtColor(obj_img, cv2.COLOR_RGB2GRAY)
    h, w = np.where(obj_gray > 0)
    xmin, xmax = np.min(w), np.max(w)
    ymin, ymax = np.min(h), np.max(h)
    return obj_img[ymin:ymax, xmin:xmax, :]

def get_dataset_list(dataset_root):
    dataset_path = []
    datasets = [os.path.join(dataset_root, i) for i in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, i))]
    cls_append_path = ['VOCdevkit/VOC2007/SegmentationClass', 'VOCdevkit/VOC2012/SegmentationClass']
    img_append_path = ['VOCdevkit/VOC2007/JPEGImages', 'VOCdevkit/VOC2012/JPEGImages']
    obj_append_path = ['VOCdevkit/VOC2007/SegmentationObject', 'VOCdevkit/VOC2012/SegmentationObject']
    for dataset in datasets:
        cls_path = os.path.join(dataset, cls_append_path[0])
        img_path = os.path.join(dataset, img_append_path[0])
        obj_path = os.path.join(dataset, obj_append_path[0])
        if not os.path.exists(cls_path):
            cls_path = os.path.join(dataset, cls_append_path[1])
            img_path = os.path.join(dataset, img_append_path[1])
            obj_path = os.path.join(dataset, obj_append_path[1])
        dataset_path.append([cls_path, img_path, obj_path])
    return dataset_path

def dataset_split(dataset):
    cls_dir, img_dir, obj_dir = dataset
    front_path = os.path.join(img_dir, '../FrontObjs')
    back_path = os.path.join(img_dir, '../BackObjs')
    if not os.path.exists(front_path):
        os.mkdir(front_path)
    if not os.path.exists(back_path):
        os.mkdir(back_path)
    kernel = np.ones((3, 3), np.uint8)
    for mask_path in glob.glob(os.path.join(obj_dir, '*.png')):
        mask_name = os.path.split(mask_path)[-1]
        img_name = mask_name.split('.')[0] + '.jpg'
        obj_mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
        obj_mask[obj_mask == BORDER['border']] = 0
        cls_mask =  cv2.cvtColor(cv2.imread(os.path.join(cls_dir, mask_name)), cv2.COLOR_BGR2RGB)
        img =  cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_name)), cv2.COLOR_BGR2RGB)
        background = img.copy()
        obj_pos = find_obj_pos(obj_mask.copy())
        for pos in obj_pos:
            obj_rgb, cls_rgb = obj_mask[pos[0], pos[1]], cls_mask[pos[0], pos[1]]
            cls_rgb = (cls_rgb[0], cls_rgb[1], cls_rgb[2])
            obj_cls = COLOR_MAP[cls_rgb]
            obj_map = cv2.inRange(obj_mask, obj_rgb, obj_rgb)
            obj_map = cv2.dilate(obj_map, kernel=kernel, iterations=1)
            obj_img = cv2.bitwise_and(img, img, mask=obj_map)
            obj_img = unpadding(obj_img)
            background = cv2.bitwise_and(background, background, mask=~obj_map)
            obj_save_path = os.path.join(front_path, obj_cls)
            if not os.path.exists(obj_save_path):
                os.mkdir(obj_save_path)
            l = len(os.listdir(obj_save_path))
            obj_img = cv2.cvtColor(obj_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(obj_save_path, '%s_%05d.png' % (img_name[:-4], l)), obj_img)
        background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
        l = len(os.listdir(back_path))
        cv2.imwrite(os.path.join(back_path, '%s_%05d.png' % (img_name[:-4], l)), background)



def main():
    dataset_root = '../../dataset'
    datasets = get_dataset_list(dataset_root)
    for dataset in datasets:
        print(dataset)
        dataset_split(dataset)


if __name__ == '__main__':
    main()
