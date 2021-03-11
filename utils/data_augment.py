# coding=utf-8
import cv2
import random
import numpy as np
import os


class RandomHorizontalFilp(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            _, w_img, _ = img.shape
            # img = np.fliplr(img)
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
        return img, bboxes


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape

            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return img, bboxes


class RandomAffine(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            # 得到可以包含所有bbox的最大bbox
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w_img, h_img))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return img, bboxes


class Resize(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """
    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes):
        h_org , w_org , _= img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes
        return image


class Mixup(object):
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        if random.random() > self.p:
            lam = np.random.beta(1.5, 1.5)
            img = lam * img_org + (1 - lam) * img_mix
            bboxes_org = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=1)
            bboxes_mix = np.concatenate(
                [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=1)
            bboxes = np.concatenate([bboxes_org, bboxes_mix])

        else:
            img = img_org
            bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)

        return img, bboxes


class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes




# # DEFINE my data augment
# class ColorChange(object):
#     def __init__(self, p=0.25):
#         self.p = p
#
#     def __call__(self, img, bboxes):
#         rnd = random.random()
#         if rnd > 3 * self.p:
#             img = [img[..., 2], img[..., 1], img[..., 0]]
#         elif rnd > 2 * self.p:
#             img = [img[..., 1], img[..., 0], img[..., 2]]
#         elif rnd > self.p:
#             img = [img[..., 0], img[..., 2], img[..., 1]]
#         else:
#             return img, bboxes
#         img = np.transpose(np.array(img), (1, 2, 0))
#         return img, bboxes
#
#
# CLASS_OBJ_PATH = {
#     'aeroplane'     : '../dataset/SPLITED_IMG/FrontObjs/aeroplane',
#     'bicycle'       : '../dataset/SPLITED_IMG/FrontObjs/bicycle',
#     'bird'          : '../dataset/SPLITED_IMG/FrontObjs/bird',
#     'boat'          : '../dataset/SPLITED_IMG/FrontObjs/boat',
#     'bottle'        : '../dataset/SPLITED_IMG/FrontObjs/bottle',
#     'bus'           : '../dataset/SPLITED_IMG/FrontObjs/bus',
#     'car'           : '../dataset/SPLITED_IMG/FrontObjs/car',
#     'cat'           : '../dataset/SPLITED_IMG/FrontObjs/cat',
#     'chair'         : '../dataset/SPLITED_IMG/FrontObjs/chair',
#     'cow'           : '../dataset/SPLITED_IMG/FrontObjs/cow',
#     'diningtable'   : '../dataset/SPLITED_IMG/FrontObjs/diningtable',
#     'dog'           : '../dataset/SPLITED_IMG/FrontObjs/dog',
#     'horse'         : '../dataset/SPLITED_IMG/FrontObjs/horse',
#     'motorbike'     : '../dataset/SPLITED_IMG/FrontObjs/motorbike',
#     'person'        : '../dataset/SPLITED_IMG/FrontObjs/person',
#     'pottedplant'   : '../dataset/SPLITED_IMG/FrontObjs/pottedplant',
#     'sheep'         : '../dataset/SPLITED_IMG/FrontObjs/sheep',
#     'sofa'          : '../dataset/SPLITED_IMG/FrontObjs/sofa',
#     'train'         : '../dataset/SPLITED_IMG/FrontObjs/train',
#     'monitor'       : '../dataset/SPLITED_IMG/FrontObjs/monitor',
# }
#
# CLASS_OBJ_ID = {
#     'aeroplane'     :  0.,
#     'bicycle'       :  1.,
#     'bird'          :  2.,
#     'boat'          :  3.,
#     'bottle'        :  4.,
#     'bus'           :  5.,
#     'car'           :  6.,
#     'cat'           :  7.,
#     'chair'         :  8.,
#     'cow'           :  9.,
#     'diningtable'   : 10.,
#     'dog'           : 11.,
#     'horse'         : 12.,
#     'motorbike'     : 13.,
#     'person'        : 14.,
#     'pottedplant'   : 15.,
#     'sheep'         : 16.,
#     'sofa'          : 17.,
#     'train'         : 18.,
#     'monitor'       : 19.,
# }
#
# BACKGROUND_PATH = {
#     'background': '../dataset/SPLITED_IMG/BackObjs',
# }
#
#
# def get_front_obj_dict():
#     front_obj_dict = {}
#     for k in CLASS_OBJ_PATH:
#         list = [os.path.join(CLASS_OBJ_PATH[k], i) for i in os.listdir(CLASS_OBJ_PATH[k])]
#         front_obj_dict[k] = list
#     return front_obj_dict
#
#
# def get_ground_dict():
#     ground_dict = {}
#     for k in BACKGROUND_PATH:
#         list = [os.path.join(BACKGROUND_PATH[k], i) for i in os.listdir(BACKGROUND_PATH[k])]
#         ground_dict[k] = list
#     return ground_dict
#
#
# FRONT_OBJ_DICT = get_front_obj_dict()
# GROUND_DICT = get_ground_dict()
# RESHAPE_BASE = [32, 64, 220]
#
#
# def calc_iou(b1, b2):
#     b1_x1, b1_y1, b1_x2, b1_y2 = b1
#     b2_x1, b2_y1, b2_x2, b2_y2 = b2
#     area_sum = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
#     left = max(b1_x1, b2_x1)
#     right = min(b1_x2, b2_x2)
#     top = max(b1_y1, b2_y1)
#     bottom = min(b1_y2, b2_y2)
#     if left > right or top > bottom:
#         intersect = 0.
#     else:
#         intersect = (right - left) * (bottom - top)
#     return (intersect / (area_sum - intersect)) * 1.
#
#
# def add_obj(img, obj_img, key, bboxes, base_scale, rand_scale):
#     h, w, _ = img.shape
#     obj_h, obj_w, _ = obj_img.shape
#     final_scale = random.choice(range(-rand_scale, rand_scale)) + base_scale
#     small_side = min(obj_h, obj_w)
#     obj_h, obj_w = int(obj_h * final_scale / small_side), int(obj_w * final_scale / small_side)
#     obj_img = cv2.resize(obj_img, (obj_w, obj_h))
#     y_max, x_max = h - obj_h, w - obj_w
#     if y_max <= 0 or x_max <= 0:
#         return img, None
#
#     if len(bboxes) == 0:
#         pos_y, pos_x = random.choice(range(y_max)), random.choice(range(x_max))
#         x1, y1, x2, y2 = pos_x, pos_y, pos_x + obj_w, pos_y + obj_h
#     else:
#         flag = True
#         TRY_NUM = 0
#         while flag:
#             flag = False
#             pos_y, pos_x = random.choice(range(y_max)), random.choice(range(x_max))
#             x1, y1, x2, y2 = pos_x, pos_y, pos_x + obj_w, pos_y + obj_h
#             for b in bboxes:
#                 thres = calc_iou([x1, y1, x2, y2], b[:4])
#                 if thres > 1e-6:
#                     flag = True
#             if TRY_NUM > 3:
#                 return img, None
#             TRY_NUM += 1
#     box = [np.float64(x1), np.float64(y1), np.float64(x2), np.float64(y2), CLASS_OBJ_ID[key]]
#     obj_mask = cv2.inRange(obj_img, (1, 1, 1), (255, 255, 255))
#     sub_img = img[y1:y2, x1:x2]
#     obj_img = cv2.bitwise_and(obj_img, obj_img, mask=obj_mask)
#     sub_img = cv2.bitwise_and(sub_img, sub_img, mask=~obj_mask)
#     img[y1:y2, x1:x2] = sub_img + obj_img
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     return img, box
#
# class RandomGenImg(object):
#     def __init__(self, p=0.1, scale_p=(0.7, 0.9)):
#         self.p = p
#         self.scale_p = scale_p
#
#     def __call__(self, img, bboxes):
#         img_ori = img.copy()
#         bboxes_ori = bboxes.copy()
#         if random.random() < self.p:
#             background_path = random.choice(GROUND_DICT['background'])
#             img = cv2.imread(background_path)
#             bboxes = []
#             h, w, _ = img.shape
#             obj_num = random.choice([1, 2, 3])
#             for _ in range(obj_num):
#                 key = random.choice(list(FRONT_OBJ_DICT.keys()))
#                 obj_path = random.choice(FRONT_OBJ_DICT[key])
#                 obj_img = cv2.imread(obj_path)
#                 scale_rnd = random.random()
#                 if scale_rnd < self.scale_p[0]:
#                     img, box = add_obj(img, obj_img, key, bboxes, 32, 10)
#                 elif scale_rnd < self.scale_p[1]:
#                     img, box = add_obj(img, obj_img, key, bboxes, 64, 20)
#                 else:
#                     img, box = add_obj(img, obj_img, key, bboxes, 220, 50)
#                 if box is not None:
#                     bboxes.append(box)
#             if len(bboxes) == 0:
#                 return img_ori, bboxes_ori
#         return img, bboxes
#
# class Mosaic(object):
#     def __init__(self):
#         pass
#
#     def __call__(self, img, bboxes):
#         return img, bboxes
