# # Copyright (c) OpenMMLab. All rights reserved.
# # Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
# try:
#     import MinkowskiEngine as ME
# except ImportError:
#     import warnings
#     warnings.warn(
#         'Please follow `getting_started.md` to install MinkowskiEngine.`')

# from mmdet3d.core import bbox3d2result
# from mmdet3d.models import DETECTORS, build_backbone, build_head, build_neck
# from .base import Base3DDetector
# import torch


# @DETECTORS.register_module()
# class MinkSingleStage3DDetector(Base3DDetector):
#     r"""Single stage detector based on MinkowskiEngine `GSDN
#     <https://arxiv.org/abs/2006.12356>`_.

#     Args:
#         backbone (dict): Config of the backbone.
#         head (dict): Config of the head.
#         voxel_size (float): Voxel size in meters.
#         neck (dict): Config of the neck.
#         train_cfg (dict, optional): Config for train stage. Defaults to None.
#         test_cfg (dict, optional): Config for test stage. Defaults to None.
#         init_cfg (dict, optional): Config for weight initialization.
#             Defaults to None.
#         pretrained (str, optional): Deprecated initialization parameter.
#             Defaults to None.
#     """

#     def __init__(self,
#                  backbone,
#                  head,
#                  voxel_size,
#                  neck=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  init_cfg=None,
#                  pretrained=None):
#         super(MinkSingleStage3DDetector, self).__init__(init_cfg)
#         self.backbone = build_backbone(backbone)
#         if neck is not None:
#             self.neck = build_neck(neck)
#         head.update(train_cfg=train_cfg)
#         head.update(test_cfg=test_cfg)
#         self.head = build_head(head)
#         self.voxel_size = voxel_size
#         self.iter = 0
#         self.iter_test = 0
#         self.init_weights()

#     def extract_feat(self, *args):
#         """Just implement @abstractmethod of BaseModule."""

#     def extract_feats(self, points):
#         """Extract features from points.

#         Args:
#             points (list[Tensor]): Raw point clouds.

#         Returns:
#             SparseTensor: Voxelized point clouds.
#         """
        
#         x = self.backbone(points)
#         if self.with_neck:
#             x = self.neck(x)
#         return x

#     def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, img_metas):
#     # def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask, img_metas):
#         """Forward of training.

#         Args:
#             points (list[Tensor]): Raw point clouds.
#             gt_bboxes (list[BaseInstance3DBoxes]): Ground truth
#                 bboxes of each sample.
#             gt_labels(list[torch.Tensor]): Labels of each sample.
#             img_metas (list[dict]): Contains scene meta infos.

#         Returns:
#             dict: Centerness, bbox and classification loss values.
#         """
#         # points = [torch.cat([p, torch.unsqueeze(inst, 1)], dim=1) for p, inst in zip(points, pts_semantic_mask)]
#         coordinates, features = ME.utils.batch_sparse_collate(
#             [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
#             device=points[0].device)
#         # coordinates_8, features_8 = ME.utils.batch_sparse_collate(
#         #     [(p[:, :3] / (self.voxel_size*8), p[:, 3:]) for p in points],
#         #     device=points[0].device)
#         # coordinates_16, features_16 = ME.utils.batch_sparse_collate(
#         #     [(p[:, :3] / (self.voxel_size*16), p[:, 3:]) for p in points],
#         #     device=points[0].device)
#         # coordinates_32, features_32 = ME.utils.batch_sparse_collate(
#         #     [(p[:, :3] / (self.voxel_size*32), p[:, 3:]) for p in points],
#         #     device=points[0].device)
#         # coordinates_64, features_64 = ME.utils.batch_sparse_collate(
#         #     [(p[:, :3] / (self.voxel_size*64), p[:, 3:]) for p in points],
#         #     device=points[0].device)
#         # coordinates_8[:, 1:] = coordinates_8[:, 1:] * 8
#         # coordinates_16[:, 1:] = coordinates_16[:, 1:] * 16
#         # coordinates_32[:, 1:] = coordinates_32[:, 1:] * 32
#         # coordinates_64[:, 1:] = coordinates_64[:, 1:] * 64
#         # features_8 = add_consistency_to_features(coordinates_8, features_8)
#         # features_16 = add_consistency_to_features(coordinates_16, features_16)
#         # features_32 = add_consistency_to_features(coordinates_32, features_32)
#         # features_64 = add_consistency_to_features(coordinates_64, features_64)

#         x = ME.SparseTensor(coordinates=coordinates, features=features[:, :3])
#         # target_8 = ME.SparseTensor(coordinates=coordinates_8, features=features_8[:, 3:])
#         # target_16 = ME.SparseTensor(coordinates=coordinates_16, features=features_16[:, 3:])
#         # target_32 = ME.SparseTensor(coordinates=coordinates_32, features=features_32[:, 3:])
#         # target_64 = ME.SparseTensor(coordinates=coordinates_64, features=features_64[:, 3:])
#         # target = [target_8, target_16, target_32, target_64]
#         x = self.extract_feats(x)
#         # for i in range(len(img_metas)):
#             # if img_metas[i]['sample_idx'] == 'scene0092_03':
#             # if img_metas[i]['sample_idx'] == '6000':
#             #     point = points[i]
#             #     point[:, 3:] *= 255
#             #     from mmdet3d.core.visualizer.show_result import _write_obj, _write_oriented_bbox_v2
#             #     _write_obj(point.detach().cpu().numpy(), f'/data/lxr/visual/assign_sunrgbd/006000_{self.iter}.obj')
#             #     self.iter += 1

#         # losses = self.head.forward_train(x, gt_bboxes_3d, gt_labels_3d, target, img_metas)
#         losses = self.head.forward_train(x, gt_bboxes_3d, gt_labels_3d, img_metas)
#         return losses

#     def simple_test(self, points, img_metas, *args, **kwargs):
#         """Test without augmentations.

#         Args:
#             points (list[torch.Tensor]): Points of each sample.
#             img_metas (list[dict]): Contains scene meta infos.

#         Returns:
#             list[dict]: Predicted 3d boxes.
#         """
#         coordinates, features = ME.utils.batch_sparse_collate(
#             [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
#             device=points[0].device)
#         x = ME.SparseTensor(coordinates=coordinates, features=features)
#         x = self.extract_feats(x)
#         # for i in range(len(img_metas)):
#             # if img_metas[i]['sample_idx'] == 'scene0011_00':
#             # if img_metas[i]['sample_idx'] == '5050':
#             #     point = points[i]
#             #     point[:, 3:] *= 255
#             #     from mmdet3d.core.visualizer.show_result import _write_obj, _write_oriented_bbox_v2
#             #     _write_obj(point.detach().cpu().numpy(), f'/data/lxr/visual/assign_sunrgbd_test/005050_{self.iter_test}.obj')
#             #     self.iter_test += 1
#         bbox_list = self.head.forward_test(x, img_metas, **kwargs)
#         bbox_results = [
#             bbox3d2result(bboxes, scores, labels)
#             for bboxes, scores, labels in bbox_list
#         ]
#         return bbox_results

#     def aug_test(self, points, img_metas, **kwargs):
#         """Test with augmentations.

#         Args:
#             points (list[list[torch.Tensor]]): Points of each sample.
#             img_metas (list[dict]): Contains scene meta infos.

#         Returns:
#             list[dict]: Predicted 3d boxes.
#         """
#         raise NotImplementedError

# import torch

# def add_consistency_to_features(coordinates, features):
#     """
#     为 features 添加一致性标记, 根据相同体素内的标签一致性判断。
    
#     :param coordinates: 体素化后的坐标 (N, 4)
#     :param features: 体素化后的特征 (N, 3+C)，前三维为坐标，后面 C 维为标签/特征
#     :return: 带有一致性标记的新 features (N, 3+C+1)
#     """
#     # 获取 features 的后 C 维（标签部分）
#     labels = features[:, 3:]  # (N, C)
    
#     # 将前三维坐标作为体素的唯一标识
#     voxel_keys = coordinates[:, :3].to(torch.float32)
    
#     # 使用 torch.unique 找出每个体素的唯一标识及其对应的索引
#     _, unique_indices = torch.unique(voxel_keys, dim=0, return_inverse=True)
    
#     # 初始化一个张量，记录每个体素的标签一致性
#     consistency_flags = torch.zeros(features.size(0), dtype=torch.float32, device=features.device)

#     # 找到所有不同体素的索引
#     for voxel_index in torch.unique(unique_indices):
#         # 对于每个体素，找到所有对应的标签
#         voxel_mask = (unique_indices == voxel_index)
#         voxel_labels = labels[voxel_mask]

#         # 检查这个体素内的标签是否一致
#         if not (voxel_labels == voxel_labels[0]).all():
#             consistency_flags[voxel_mask] = 1

#     # 将一致性标记添加到 features
#     consistency_flags = consistency_flags.unsqueeze(1)
#     new_features = torch.cat([features, consistency_flags], dim=1)

#     return new_features

