# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print ('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print ('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    scaled_npos = {'small':0, 'medium':0, 'large':0}
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        # bbox: x1,y1,x2,y2  difficult: *    det: is detected
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
        for box in bbox:
            x1, y1, x2, y2 = box
            box_area = (x2-x1) * (y2-y1)
            if box_area <= 32*32:
                scaled_npos['small'] += 1
            elif box_area <= 96*96:
                scaled_npos['medium'] += 1
            else:
                scaled_npos['large'] += 1

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    scaled_tp = {'small': np.zeros(nd), 'medium': np.zeros(nd), 'large': np.zeros(nd)}
    scaled_fp = {'small': np.zeros(nd), 'medium': np.zeros(nd), 'large': np.zeros(nd)}
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            x1, y1, x2, y2 = R['bbox'][jmax]
            gt_area = (x2 - x1) * (y2 - y1)
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                    if gt_area <= 32*32:
                        scaled_tp['small'][d] = 1.
                    elif gt_area <= 96*96:
                        scaled_tp['medium'][d] = 1.
                    else:
                        scaled_tp['large'][d] = 1.
                else:
                    fp[d] = 1.
                    if gt_area <= 32*32:
                        scaled_fp['small'][d] = 1.
                    elif gt_area <= 96*96:
                        scaled_fp['medium'][d] = 1.
                    else:
                        scaled_fp['large'][d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    scaled_fp = {'small': np.cumsum(scaled_fp['small']),
                 'medium': np.cumsum(scaled_fp['medium']),
                 'large': np.cumsum(scaled_fp['large'])}
    tp = np.cumsum(tp)
    scaled_tp = {'small': np.cumsum(scaled_tp['small']),
                 'medium': np.cumsum(scaled_tp['medium']),
                 'large': np.cumsum(scaled_tp['large'])}
    rec = tp / float(npos)
    scaled_rec = {'small': scaled_tp['small'] / float(scaled_npos['small']),
                  'medium': scaled_tp['medium'] / float(scaled_npos['medium']),
                  'large': scaled_tp['large'] / float(scaled_npos['large']),}
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    scaled_prec = {
        'small': scaled_tp['small'] / np.maximum(scaled_tp['small'] + scaled_fp['small'], np.finfo(np.float64).eps),
        'medium': scaled_tp['medium'] / np.maximum(scaled_tp['medium'] + scaled_fp['medium'], np.finfo(np.float64).eps),
        'large': scaled_tp['large'] / np.maximum(scaled_tp['large'] + scaled_fp['large'], np.finfo(np.float64).eps),
    }
    ap = voc_ap(rec, prec, use_07_metric)
    scaled_ap = {'small': voc_ap(scaled_rec['small'], scaled_prec['small'], use_07_metric),
                 'medium': voc_ap(scaled_rec['medium'], scaled_prec['medium'], use_07_metric),
                 'large': voc_ap(scaled_rec['large'], scaled_prec['large'], use_07_metric),}
    return rec, prec, ap, scaled_ap
