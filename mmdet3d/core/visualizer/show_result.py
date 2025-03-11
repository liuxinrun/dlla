# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import torch
from mmdet3d.core.bbox.structures.utils import rotation_3d_in_axis
import random

from .image_vis import (draw_camera_bbox3d_on_img, draw_depth_bbox3d_on_img,
                        draw_lidar_bbox3d_on_img)

color_scannet = [[0, 0, 102],
          [102, 102, 153],
          [153, 0, 255],
          [102, 0, 102],
          [204, 0, 153],
          [102, 0, 51],
          [153, 51, 51],
          [255, 0, 0],
          [204, 102, 0],
          [102, 51, 0],
          [153, 153, 102],
          [255, 255, 0],
          [102, 153, 0],
          [102, 255, 51],
          [0, 51, 0],
          [102, 153, 153],
          [0, 51, 102],
          [51, 102, 204],
          [0, 0, 255]]

from colorsys import hls_to_rgb

def generate_hsl_colors(num_classes=200):
    colors = []
    for i in range(num_classes):
        hue = i / num_classes  # 色相从 0 到 1 均匀分布

        # 随机改变亮度和饱和度以增加对比度
        lightness = 0.5 + (random.random() * 0.3)  # 在 0.5 到 0.8 之间随机取值
        saturation = 0.7 + (random.random() * 0.3)  # 在 0.7 到 1.0 之间随机取值

        # 转换为 RGB 值
        rgb = hls_to_rgb(hue, lightness, saturation)
        rgb = tuple(int(c * 255) for c in rgb)  # 转换为 RGB 0-255 范围
        colors.append(rgb)

    # 随机打乱颜色，防止相邻颜色太接近
    random.shuffle(colors)
    return colors
colors_scannet200 = [(154, 172, 238), (136, 242, 42), (205, 229, 109), (62, 241, 208), (244, 158, 56), (236, 66, 229), (132, 252, 158), (233, 172, 70), (192, 48, 242), (133, 233, 128), (41, 132, 239), (234, 118, 194), (240, 110, 192), (39, 254, 170), (67, 240, 23), (249, 59, 156), (243, 58, 224), (132, 228, 98), (248, 130, 151), (153, 229, 67), (248, 86, 139), (61, 235, 177), (33, 35, 249), (191, 137, 234), (235, 64, 47), (216, 124, 244), (240, 65, 27), (49, 154, 239), (49, 63, 245), (248, 169, 155), (49, 39, 238), (48, 204, 238), (55, 228, 252), (97, 240, 154), (40, 230, 249), (92, 228, 216), (227, 92, 149), (157, 243, 116), (89, 248, 24), (91, 241, 106), (233, 60, 76), (227, 243, 145), (67, 134, 222), (230, 128, 78), (80, 94, 227), (149, 216, 241), (141, 125, 240), (158, 32, 235), (229, 132, 101), (134, 237, 163), (171, 224, 40), (240, 237, 162), (249, 152, 110), (233, 189, 34), (251, 187, 140), (141, 145, 243), (107, 245, 249), (241, 135, 121), (246, 151, 254), (229, 59, 136), (79, 250, 81), (246, 123, 223), (39, 251, 92), (171, 114, 243), (244, 110, 162), (199, 130, 247), (72, 69, 238), (133, 160, 240), (241, 144, 219), (116, 252, 104), (119, 238, 123), (223, 129, 32), (224, 251, 147), (236, 141, 241), (204, 240, 158), (248, 250, 140), (96, 185, 242), (243, 111, 186), (163, 83, 234), (228, 238, 146), (246, 89, 98), (181, 36, 250), (147, 233, 239), (80, 195, 252), (165, 245, 115), (37, 234, 145), (118, 204, 240), (230, 174, 91), (82, 247, 114), (254, 137, 239), (109, 38, 225), (236, 140, 209), (254, 159, 80), (248, 159, 247), (244, 242, 114), (41, 198, 226), (52, 225, 142), (234, 180, 59), (211, 254, 140), (187, 76, 248), (246, 157, 99), (135, 244, 238), (241, 37, 110), (44, 70, 248), (103, 239, 31), (243, 58, 196), (251, 243, 134), (239, 149, 250), (216, 53, 249), (152, 251, 74), (236, 106, 58), (248, 216, 120), (238, 156, 176), (64, 98, 246), (228, 81, 72), (138, 88, 231), (134, 237, 222), (114, 174, 238), (62, 229, 28), (247, 139, 191), (130, 249, 224), (149, 202, 251), (145, 237, 212), (47, 98, 228), (247, 220, 106), (88, 245, 62), (139, 237, 137), (148, 89, 232), (250, 132, 203), (241, 251, 47), (253, 144, 115), (250, 31, 77), (205, 247, 116), (192, 236, 129), (116, 231, 159), (15, 110, 251), (226, 240, 74), (98, 122, 251), (114, 238, 194), (98, 43, 252), (244, 132, 132), (215, 157, 239), (239, 213, 146), (216, 233, 110), (146, 120, 234), (48, 25, 242), (254, 205, 140), (73, 223, 138), (150, 208, 242), (244, 56, 248), (127, 170, 253), (80, 241, 202), (253, 3, 176), (141, 245, 37), (88, 247, 109), (63, 229, 119), (253, 109, 113), (253, 73, 225), (111, 46, 251), (129, 246, 94), (197, 154, 240), (253, 235, 114), (124, 94, 245), (250, 238, 131), (241, 209, 146), (101, 72, 244), (254, 137, 134), (245, 162, 187), (234, 165, 101), (82, 235, 106), (196, 35, 237), (223, 252, 128), (108, 231, 23), (27, 231, 170), (243, 113, 183), (207, 252, 122), (86, 247, 165), (178, 112, 253), (24, 232, 88), (70, 242, 149), (13, 247, 247), (243, 138, 151), (43, 200, 250), (112, 213, 240), (234, 162, 122), (253, 96, 120), (237, 216, 107), (47, 106, 238), (93, 251, 185), (75, 62, 223), (254, 134, 166), (109, 243, 227), (42, 155, 237), (160, 126, 244), (47, 219, 59), (135, 244, 121), (143, 177, 236), (145, 243, 240), (250, 126, 242), (223, 29, 254)]

def box_to_corners(box):
    dims = box[:, 3:6]
    corners_norm = torch.from_numpy(
        np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
            device=dims.device, dtype=dims.dtype)

    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0.5]
    corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0.5])
    corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    if box.shape[1] == 6:
        theta = torch.zeros(corners.shape[0], device=corners.device)
    else:
        theta = box[:, 6]

    corners = rotation_3d_in_axis(
        corners, theta, axis=1)
    corners += box[:, :3].view(-1, 1, 3)
    return corners

def _write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def _write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes.

    Args:
        scene_bbox(list[ndarray] or ndarray): xyz pos of center and
            3 lengths (x_size, y_size, z_size) and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename(str): Filename.
    """

    def heading2rotmat(heading_angle):
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    if len(scene_bbox) == 0:
        scene_bbox = np.zeros((1, 7))
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to obj file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='obj')

    return


def _write_oriented_bbox_v2(corners, labels, out_filename, num_class=18):
    if num_class <= 10:
        colors = np.multiply([plt.cm.get_cmap('tab10', num_class)(i)[:3] for i in range(num_class)], 255).astype(np.uint8).tolist()
    elif num_class <=20:
        colors = np.multiply([plt.cm.get_cmap('tab20', num_class)(i)[:3] for i in range(num_class)], 255).astype(np.uint8).tolist()
    else:
        colors = colors_scannet200
    # colors = np.multiply([
    #     plt.cm.get_cmap('nipy_spectral', num_class)((i * 5 + 11) % num_class + 1)[:3] for i in range(num_class)
    # ], 255).astype(np.uint8).tolist()
    # colors = np.array(color_scannet).astype(np.uint8).tolist()
    with open(out_filename, 'w') as file:
        for i, (corner, label) in enumerate(zip(corners, labels)):
            c = colors[label]
            for p in corner:
                # file.write(f'v {p[0]} {p[1]} {p[2]}\n')
                file.write(f'v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n')
            j = i * 8 + 1
            for k in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                      [2, 3, 7, 6], [3, 0, 4, 7], [1, 2, 6, 5]]:
                file.write('f')
                for l in k:
                    # file.write(f' {j + l} {c[0]} {c[1]} {c[2]}')
                    file.write(f' {j + l}')
                file.write('\n')


def show_result(points,
                gt_bboxes,
                pred_bboxes,
                out_dir,
                filename,
                show=False,
                snapshot=False,
                pred_labels=None):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results.
            Defaults to False.
        pred_labels (np.ndarray, optional): Predicted labels of boxes.
            Defaults to None.
    """
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:
        from .open3d_vis import Visualizer

        vis = Visualizer(points)
        if pred_bboxes is not None:
            if pred_labels is None:
                vis.add_bboxes(bbox3d=pred_bboxes)
            else:
                palette = np.random.randint(
                    0, 255, size=(pred_labels.max() + 1, 3)) / 256
                labelDict = {}
                for j in range(len(pred_labels)):
                    i = int(pred_labels[j].numpy())
                    if labelDict.get(i) is None:
                        labelDict[i] = []
                    labelDict[i].append(pred_bboxes[j])
                for i in labelDict:
                    vis.add_bboxes(
                        bbox3d=np.array(labelDict[i]),
                        bbox_color=palette[i],
                        points_in_box_color=palette[i])

        if gt_bboxes is not None:
            vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_bboxes is not None:
        # bottom center to gravity center
        gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2

        _write_oriented_bbox(gt_bboxes,
                             osp.join(result_path, f'{filename}_gt.obj'))

    if pred_bboxes is not None:
        # bottom center to gravity center
        pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2

        _write_oriented_bbox(pred_bboxes,
                             osp.join(result_path, f'{filename}_pred.obj'))


def show_result_v2(points,
                   gt_corners,
                   gt_labels,
                   pred_corners,
                   pred_labels,
                   out_dir,
                   filename):
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_corners is not None:
        _write_oriented_bbox_v2(gt_corners, gt_labels,
                                osp.join(result_path, f'{filename}_gt.obj'))

    if pred_corners is not None:
        _write_oriented_bbox_v2(pred_corners, pred_labels,
                                osp.join(result_path, f'{filename}_pred.obj'))


def show_seg_result(points,
                    gt_seg,
                    pred_seg,
                    out_dir,
                    filename,
                    palette,
                    ignore_index=None,
                    show=False,
                    snapshot=False):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_seg (np.ndarray): Ground truth segmentation mask.
        pred_seg (np.ndarray): Predicted segmentation mask.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        palette (np.ndarray): Mapping between class labels and colors.
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. Defaults to None.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results.
            Defaults to False.
    """
    # we need 3D coordinates to visualize segmentation mask
    if gt_seg is not None or pred_seg is not None:
        assert points is not None, \
            '3D coordinates are required for segmentation visualization'

    # filter out ignored points
    if gt_seg is not None and ignore_index is not None:
        if points is not None:
            points = points[gt_seg != ignore_index]
        if pred_seg is not None:
            pred_seg = pred_seg[gt_seg != ignore_index]
        gt_seg = gt_seg[gt_seg != ignore_index]

    if gt_seg is not None:
        gt_seg_color = palette[gt_seg]
        gt_seg_color = np.concatenate([points[:, :3], gt_seg_color], axis=1)
    if pred_seg is not None:
        pred_seg_color = palette[pred_seg]
        pred_seg_color = np.concatenate([points[:, :3], pred_seg_color],
                                        axis=1)

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    # online visualization of segmentation mask
    # we show three masks in a row, scene_points, gt_mask, pred_mask
    if show:
        from .open3d_vis import Visualizer
        mode = 'xyzrgb' if points.shape[1] == 6 else 'xyz'
        vis = Visualizer(points, mode=mode)
        if gt_seg is not None:
            vis.add_seg_mask(gt_seg_color)
        if pred_seg is not None:
            vis.add_seg_mask(pred_seg_color)
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_seg is not None:
        _write_obj(gt_seg_color, osp.join(result_path, f'{filename}_gt.obj'))

    if pred_seg is not None:
        _write_obj(pred_seg_color, osp.join(result_path,
                                            f'{filename}_pred.obj'))


def show_multi_modality_result(img,
                               gt_bboxes,
                               pred_bboxes,
                               proj_mat,
                               out_dir,
                               filename,
                               box_mode='lidar',
                               img_metas=None,
                               show=False,
                               gt_bbox_color=(61, 102, 255),
                               pred_bbox_color=(241, 101, 72)):
    """Convert multi-modality detection results into 2D results.

    Project the predicted 3D bbox to 2D image plane and visualize them.

    Args:
        img (np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str, optional): Coordinate system the boxes are in.
            Should be one of 'depth', 'lidar' and 'camera'.
            Defaults to 'lidar'.
        img_metas (dict, optional): Used in projecting depth bbox.
            Defaults to None.
        show (bool, optional): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int), optional): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61).
        pred_bbox_color (str or tuple(int), optional): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241).
    """
    if box_mode == 'depth':
        draw_bbox = draw_depth_bbox3d_on_img
    elif box_mode == 'lidar':
        draw_bbox = draw_lidar_bbox3d_on_img
    elif box_mode == 'camera':
        draw_bbox = draw_camera_bbox3d_on_img
    else:
        raise NotImplementedError(f'unsupported box mode {box_mode}')

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:
        show_img = img.copy()
        if gt_bboxes is not None:
            show_img = draw_bbox(
                gt_bboxes, show_img, proj_mat, img_metas, color=gt_bbox_color)
        if pred_bboxes is not None:
            show_img = draw_bbox(
                pred_bboxes,
                show_img,
                proj_mat,
                img_metas,
                color=pred_bbox_color)
        mmcv.imshow(show_img, win_name='project_bbox3d_img', wait_time=0)

    if img is not None:
        mmcv.imwrite(img, osp.join(result_path, f'{filename}_img.png'))

    if gt_bboxes is not None:
        gt_img = draw_bbox(
            gt_bboxes, img, proj_mat, img_metas, color=gt_bbox_color)
        mmcv.imwrite(gt_img, osp.join(result_path, f'{filename}_gt.png'))

    if pred_bboxes is not None:
        pred_img = draw_bbox(
            pred_bboxes, img, proj_mat, img_metas, color=pred_bbox_color)
        mmcv.imwrite(pred_img, osp.join(result_path, f'{filename}_pred.png'))
