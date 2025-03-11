# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import warnings
import numpy as np

def find_latest_checkpoint(path, suffix='pth'):
    """Find the latest checkpoint from the working directory. This function is
    copied from mmdetection.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path

def nms_3d_faster_samecls(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    z1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    z2 = boxes[:, 5]
    score = boxes[:, 6]
    cls = boxes[:, 7]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    I = np.argsort(score)
    pick = []
    while I.size != 0:
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[: last - 1]])
        yy1 = np.maximum(y1[i], y1[I[: last - 1]])
        zz1 = np.maximum(z1[i], z1[I[: last - 1]])
        xx2 = np.minimum(x2[i], x2[I[: last - 1]])
        yy2 = np.minimum(y2[i], y2[I[: last - 1]])
        zz2 = np.minimum(z2[i], z2[I[: last - 1]])
        cls1 = cls[i]
        cls2 = cls[I[: last - 1]]

        l = np.maximum(0, xx2 - xx1)
        w = np.maximum(0, yy2 - yy1)
        h = np.maximum(0, zz2 - zz1)

        if old_type:
            o = (l * w * h) / area[I[: last - 1]]
        else:
            inter = l * w * h
            o = inter / (area[i] + area[I[: last - 1]] - inter)
        o = o * (cls1 == cls2)

        I = np.delete(
            I, np.concatenate(([last - 1], np.where(o > overlap_threshold)[0]))
        )

    return pick