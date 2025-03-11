# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/dense_heads/fcaf3d_neck_with_head.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner.base_module import BaseModule
from torch import nn
import torch.nn.functional as F
from mmdet3d.models import HEADS, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner
from mmdet3d.models.losses.rotated_iou_loss import diff_diou_rotated_3d
from mmdet3d.core.bbox.iou_calculators import axis_aligned_bbox_overlaps_3d

def attention(q, k, v, nhead, channel, mask=None):
    q_n, k_n, v_n = [x.view(-1, nhead, channel//nhead).transpose(0,1) for x in (q, k, v)]
    d_k = q_n.size(-1)
    scores = torch.matmul(q_n, k_n.transpose(-2, -1)) / (d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    att = F.softmax(scores, dim=-1)
    q_feature = torch.matmul(att, v_n).transpose(0,1).contiguous().view(-1, nhead*d_k)
    q_feature += q
    return q_feature


@HEADS.register_module()
class DLLA_tr3d_Head(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 out_channels,
                 n_reg_outs,
                 voxel_size,
                 mode,
                 pro_enc_repeat=2,
                 noise_scale=1.2,
                 label2level=None,
                 assign_type="volume",
                 volume_threshold=27,
                 top_pts_threshold=6,
                 bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 localization_loss=dict(type='CrossEntropyLoss', use_sigmoid=True),
                 train_cfg=None,
                 test_cfg=None):
        super(DLLA_tr3d_Head, self).__init__()
        self.voxel_size = voxel_size
        self.label2level = label2level
        self.top_pts_threshold = top_pts_threshold
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.localization_loss = build_loss(localization_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assign_type = assign_type
        self.volume_threshold = volume_threshold
        self.channel = 256
        self.nhead = 4
        self.mode = mode
        self.dn = True
        self.noise_scale = noise_scale
        self.num_classes = n_classes
        self.pro_enc_repeat = pro_enc_repeat
        self.iter_test = 0
        self._init_layers(in_channels[1:], out_channels, n_reg_outs, n_classes, pro_enc_repeat)

    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes, pro_enc_repeat=2):
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self._make_up_block(in_channels[i], in_channels[i - 1]))
            if i < len(in_channels) - 1:
                self.__setattr__(
                    f'lateral_block_{i}',
                    self._make_block(in_channels[i], in_channels[i]))
                self.__setattr__(
                    f'out_cls_block_{i}',
                    self._make_block(in_channels[i], out_channels))
                self.__setattr__(
                    f'out_reg_block_{i}',
                    self._make_block(in_channels[i], out_channels))
        self.bbox_conv = ME.MinkowskiConvolution(
            out_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.localization_conv = ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3)
        for i in range(pro_enc_repeat):
            if i == 0:
                self.__setattr__(f'qkv{i}', nn.Linear((6+n_classes)*3, self.channel*3))
            else:
                self.__setattr__(f'qkv{i}', nn.Linear(self.channel*3, self.channel*3))
            self.__setattr__(f'pe_linear{i}', nn.Linear(3, self.nhead))
        self.q_linear = nn.Linear(self.channel, self.channel)
        self.k_linear = nn.Linear((6+n_classes), self.channel)

    def init_weights(self):
        for n, m in self.named_modules():
            if ('bbox_conv' not in n) and ('cls_conv' not in n):
                if isinstance(m, ME.MinkowskiConvolution):
                    ME.utils.kaiming_normal_(
                        m.kernel, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                for p in m.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))
        nn.init.normal_(self.localization_conv.kernel, std=.01)
    

    @staticmethod
    def _make_block(in_channels, out_channels):
        """Construct Conv-Norm-Act block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            torch.nn.Module: With corresponding layers.
        """
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU())

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        conv = ME.MinkowskiGenerativeConvolutionTranspose 
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    def forward(self, x):
        """Forward pass.

        Args:
            x (list[Tensor]): Features from the backbone.

        Returns:
            list[list[Tensor]]: Predictions of the head.
        """
        x = x[1:]
        inputs = x
        x = inputs[-1]
        bbox_preds, cls_preds, localization_preds, points = [], [], [], []
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self.__getattr__(f'lateral_block_{i}')(x)
                out_cls = self.__getattr__(f'out_cls_block_{i}')(x)
                out_reg = self.__getattr__(f'out_reg_block_{i}')(x)
                _, cls_pred, localization_pred, point = self._forward_single(out_cls)
                bbox_pred, _, _, _ = self._forward_single(out_reg)
                bbox_preds.append(bbox_pred)
                cls_preds.append(cls_pred)
                localization_preds.append(localization_pred)
                points.append(point)
        return bbox_preds[::-1], cls_preds[::-1], localization_preds[::-1], points[::-1]


    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        bbox_preds, cls_preds, localization_preds, points = self(x)
        return self._loss(bbox_preds, cls_preds, localization_preds, points,
                          gt_bboxes, gt_labels, img_metas)


    def forward_test(self, x, img_metas, **kwargs):
        bbox_preds, cls_preds, localization_preds, points = self(x)
        for i in range(len(img_metas)):
            sample_idx = img_metas[i]["sample_idx"]
            if sample_idx == 'scene0015_00' or sample_idx == 5500:
                self.iter_test += 1
        return self._get_bboxes(bbox_preds, cls_preds, localization_preds, points, img_metas)

     # per level
    def _forward_single(self, x):
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
        cls_pred = self.cls_conv(x).features
        localization_pred = self.localization_conv(x).features

        bbox_preds, cls_preds, localization_preds, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            localization_preds.append(localization_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        return bbox_preds, cls_preds, localization_preds, points

    # per scene
    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     localization_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        if self.mode == "learn":
            if self.iter_test > 0:
                assigned_ids, l2_loss_inter, l2_loss_intra_axis_align, dn_loss = self._get_targets(points, bbox_preds, cls_preds, localization_preds, gt_bboxes, gt_labels, img_meta)
            else:
                with torch.no_grad():
                    assigned_ids = self._get_targets(points, bbox_preds, cls_preds, localization_preds, gt_bboxes, gt_labels, img_meta)
                    loss_weight = torch.zeros_like(assigned_ids).float()
                    l2_loss_inter, l2_loss_intra_axis_align, dn_loss = None, None, None
        else:
            with torch.no_grad():
                assigned_ids = self._get_targets(points, bbox_preds, cls_preds, localization_preds, gt_bboxes, gt_labels, img_meta)
                l2_loss_inter, l2_loss_intra_axis_align, dn_loss = None, None, None
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        localization_preds = torch.cat(localization_preds)
        points = torch.cat(points)

        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0
        
        cluster_loss_inter = l2_loss_inter 
        cluster_loss_intra = l2_loss_intra_axis_align

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)

        cls_loss = self.cls_loss(cls_preds, cls_targets)
        # bbox loss
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))
            l1_loss = F.l1_loss(self._bbox_pred_to_bbox(pos_points, pos_bbox_preds), pos_bbox_targets,
                                reduction='none').sum(-1)
            #localization_loss
            pos_localization_preds = localization_preds[pos_mask]
            iou_preds_to_targets = axis_aligned_bbox_overlaps_3d(self._bbox_to_loss(self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                                    self._bbox_to_loss(pos_bbox_targets))
            iou_preds_to_targets = torch.diag(iou_preds_to_targets)
            pos_localization_targets = torch.where(iou_preds_to_targets > 0.3, iou_preds_to_targets, 0).unsqueeze(1)

            localization_loss = self.localization_loss(pos_localization_preds, pos_localization_targets, avg_factor=pos_mask.sum())
        else:
            bbox_loss, l1_loss, localization_loss = None, None, None
        return bbox_loss, l1_loss, cls_loss, localization_loss, cluster_loss_inter, cluster_loss_intra, dn_loss, pos_mask

    def _loss(self, bbox_preds, cls_preds, localization_preds, points, gt_bboxes, gt_labels, img_metas):
        bbox_losses, l1_losses, cls_losses, localization_losses, cluster_losses_inter, cluster_losses_intra, dn_losses, pos_masks = [], [], [], [], [], [], [], []
        for i in range(len(img_metas)):
            bbox_loss, l1_loss, cls_loss, localization_loss, cluster_loss_inter, cluster_loss_intra, dn_loss, pos_mask = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                localization_preds=[x[i] for x in localization_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],)
            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
                l1_losses.append(l1_loss)
                localization_losses.append(localization_loss)
                cluster_losses_inter.append(cluster_loss_inter)
                cluster_losses_intra.append(cluster_loss_intra)
            if dn_loss is not None:
                dn_losses.append(dn_loss)
            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)
        filtered_cluster_losses_inter = [loss for loss in cluster_losses_inter if loss is not None]
        if self.mode == "learn":
            if self.iter_test > 0 and filtered_cluster_losses_inter:
                return dict(
                bbox_loss=torch.mean(torch.cat(bbox_losses)),
                cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)),
                localization_loss= 0.1*torch.mean(torch.stack(localization_losses)),
                cluster_loss_inter = 0.5*torch.sum(torch.cat(cluster_losses_inter)) / len(img_metas),
                cluster_loss_intra = torch.sum(torch.cat(cluster_losses_intra)) / len(img_metas),
                dn_loss = torch.sum(torch.cat(dn_losses)) / len(img_metas),
                )
            else:
                return dict(
                bbox_loss=torch.mean(torch.cat(bbox_losses)),
                cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)),
                localization_loss= 0.1*torch.mean(torch.stack(localization_losses)),
                )
        else:
            return dict(
                bbox_loss=torch.mean(torch.cat(bbox_losses)),
                cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)),
                )

    def _get_bboxes_single(self, bbox_preds, cls_preds, localization_preds, points, img_meta):
        bbox_preds = torch.cat(bbox_preds)
        theta = 0.5
        scores = pow(torch.cat(localization_preds).sigmoid(), theta) * pow(torch.cat(cls_preds).sigmoid(), (1 - theta))
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)
        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        boxes, scores, labels = self._nms(boxes, scores, img_meta)
        return boxes, scores, labels

    def _get_bboxes(self, bbox_preds, cls_preds, localization_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                localization_preds=[x[i] for x in localization_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.

        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox
        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.

        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
        y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
        z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 0] + bbox_pred[:, 1] + \
            bbox_pred[:, 2] + bbox_pred[:, 3]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        if self.test_cfg.get("class_agnostic", False):
            scores, labels = torch.max(scores, dim=-1)
            ids = scores[:] > self.test_cfg.score_thr
            if ids.any():
                class_scores = scores[ids]
                class_bboxes = bboxes[ids]
                class_labels = labels[ids]
                if yaw_flag:
                    nms_function = nms3d
                else:
                    class_bboxes = torch.cat(
                        (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                        dim=1)
                    nms_function = nms3d_normal

                nms_ids = nms_function(class_bboxes, class_scores,
                                    self.test_cfg.iou_thr)
                nms_bboxes.append(class_bboxes[nms_ids])
                nms_scores.append(class_scores[nms_ids])
                nms_labels.append(class_labels[nms_ids])
        else:
            for i in range(n_classes):
                ids = scores[:, i] > self.test_cfg.score_thr
                if not ids.any():
                    continue

                class_scores = scores[ids, i]
                class_bboxes = bboxes[ids]
                if yaw_flag:
                    nms_function = nms3d
                else:
                    class_bboxes = torch.cat(
                        (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                        dim=1)
                    nms_function = nms3d_normal

                nms_ids = nms_function(class_bboxes, class_scores,
                                    self.test_cfg.iou_thr)
                nms_bboxes.append(class_bboxes[nms_ids])
                nms_scores.append(class_scores[nms_ids])
                nms_labels.append(
                    bboxes.new_full(
                        class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels


    def _get_targets(self, points, bbox_preds, cls_preds, localization_preds,  gt_bboxes, gt_labels, img_meta):
        # -> object id or -1 for each point
        float_max = points[0].new_tensor(1e8)
        float_min = -1 * float_max
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        points = torch.cat(points)
        bbox_preds = torch.cat(bbox_preds)
        unsigmoid_cls_pred = torch.cat(cls_preds)
        cls_preds = torch.cat(cls_preds).sigmoid()
        localization_preds = torch.cat(localization_preds).sigmoid()
        bbox_preds = self._bbox_pred_to_bbox(points, bbox_preds)[:, :6]
        center_preds = bbox_preds[:, :3]
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        shape = (n_points, n_boxes, -1)
        points2boxes = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        if self.assign_type == 'volume':
            bbox_state = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            label2level = torch.zeros([len(gt_bboxes),]).to(points.device)
            downsample_times = [4,3]
            for n in range(len(gt_bboxes)):
                bbox_volume = bbox_state[n][3] * bbox_state[n][4] * bbox_state[n][5]
                for i in range(len(downsample_times)):
                    if bbox_volume > self.volume_threshold * (self.voxel_size * 2 ** downsample_times[i]) ** 3:
                        label2level[n] = 1 - i
                        break 
              
        if len(gt_labels) == 0 and self.mode == "learn" and self.iter_test > 0:
            return gt_labels.new_full((n_points,), -1), None, None, None
        elif len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)

        # if self.mode == "tr3d":
        if self.iter_test < 1:
            boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)

            # condition 1: fix level for label
            if self.assign_type != 'volume':
                label2level = gt_labels.new_tensor(self.label2level)
                label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
            else:
                label_levels = label2level.unsqueeze(0).expand(n_points, n_boxes)
            point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
            level_condition = label_levels == point_levels

            # condition 2: keep topk location per box by center distance
            center = boxes[..., :3]
            center_distances = torch.sum(torch.pow(center - points2boxes, 2), dim=-1)
            center_distances = torch.where(level_condition, center_distances, float_max)
            topk_distances = torch.topk(center_distances,
                                        min(self.top_pts_threshold + 1, len(center_distances)),
                                        largest=False, dim=0).values[-1]
            topk_condition = center_distances < topk_distances.unsqueeze(0)

            # condition 3.0: only closest object to point
            center_distances = torch.sum(torch.pow(center - points2boxes, 2), dim=-1)
            _, min_inds_ = center_distances.min(dim=1)

            # condition 3: min center distance to box per point
            center_distances = torch.where(topk_condition, center_distances, float_max)
            min_values, min_ids = center_distances.min(dim=1)
            min_inds = torch.where(min_values < float_max, min_ids, -1)
            min_inds = torch.where(min_inds == min_inds_, min_ids, -1)
            boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            gt_boxes = boxes[:, :6].to(points.device)

            return min_inds
        else:
            boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            gt_boxes = boxes[:, :6].to(points.device)
            boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
            pred_xyzxyz = torch.cat((bbox_preds[:, :3] - bbox_preds[:, 3:] / 2, bbox_preds[:, :3] + bbox_preds[:, 3:] / 2), dim=-1)
            gt_xyzxyz = torch.cat((gt_boxes[:, :3] - gt_boxes[:, 3:] / 2, gt_boxes[:, :3] + gt_boxes[:, 3:] / 2), dim=-1)
            mask_matrix = gt_labels[:, None] != gt_labels
            # condition 1: fix level for label
            if self.assign_type != 'volume':
                label2level = gt_labels.new_tensor(self.label2level)
                label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
            else:
                label_levels = label2level.unsqueeze(0).expand(n_points, n_boxes)
            point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
            level_condition = label_levels == point_levels

            # condition 2: keep topk location per box by center distance
            center = boxes[..., :3]
            center_distances = torch.sum(torch.pow(center - points2boxes, 2), dim=-1)
            center_distances = torch.where(level_condition, center_distances, float_max)
            top3k_distances = torch.topk(center_distances,
                                        min(self.top_pts_threshold + 1, len(center_distances)),
                                        largest=False, dim=0).values[-1]
            top3k_condition = center_distances < top3k_distances.unsqueeze(0)

            # condition 3.0: only closest object to point
            center_distances = torch.sum(torch.pow(center - points2boxes, 2), dim=-1)
            _, min_3k_inds_ = center_distances.min(dim=1)

            # condition 3: min center distance to box per point
            center_distances = torch.where(top3k_condition, center_distances, float_max)
            min_values, min_ids = center_distances.min(dim=1)
            min_3k_inds = torch.where(min_values < float_max, min_ids, -1)
            min_3k_inds = torch.where(min_3k_inds == min_3k_inds_, min_ids, -1)
            selected_points = points[min_3k_inds >= 0]
            selected_bbox_preds = bbox_preds[min_3k_inds >= 0]
            selected_pred_cls = cls_preds[min_3k_inds >= 0]
            selected_level_condition = level_condition[min_3k_inds >= 0]
                
            q_feature = torch.cat((selected_bbox_preds, selected_pred_cls), -1)
            gt_cls = torch.eye(self.num_classes)[gt_labels].to(bbox_preds.device)
            gt_feature = torch.cat((gt_boxes, gt_cls), -1)
            for i in range(self.pro_enc_repeat):
                qkv = self.__getattr__(f'qkv{i}')(torch.cat((q_feature, q_feature, q_feature),-1))
                q, k, v = qkv[:, :self.channel], qkv[:, self.channel:2*self.channel], qkv[:, 2*self.channel:]
                q_feature = attention(q, k, v, self.nhead, self.channel)
            q_feature = self.q_linear(q_feature)
            k_feature = self.k_linear(gt_feature)
            
            if self.dn:
                 scalars = 1
                 dn_losses = torch.zeros_like(nn.PairwiseDistance(p=2)(k_feature, k_feature.detach()))
                 for i in range(scalars):
                    scalar = 1 # scalar set 1
                    known_boxes = gt_boxes.repeat(scalar, 1)
                    known_labels = gt_labels.repeat(scalar, 1).view(-1)
                    known_labels_expaned = known_labels.clone()
                    known_box_expand = known_boxes.clone()
                    if self.noise_scale > 0:
                            p = torch.rand_like(known_labels_expaned.float())
                            chosen_indice = torch.nonzero(p < (0.5*self.noise_scale)).view(-1)  # half of box prob
                            new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                            known_labels_expaned.scatter_(0, chosen_indice, new_label)
                            diff = torch.zeros_like(known_box_expand)
                            diff[:, :3] = known_box_expand[:, 3:] / 2
                            diff[:, 3:] = known_box_expand[:, 3:]
                            known_box_expand += torch.mul((torch.rand_like(known_box_expand) * 2 - 1.0),
                                                        diff).cuda() * self.noise_scale
                            known_box_expand[:, 3:] = known_box_expand[:, 3:].clamp(min=0.0)
                    single_pad = int(n_boxes)
                    pad_size = int(single_pad * scalar)
                    padding_label = torch.zeros(pad_size, self.num_classes).cuda()
                    padding_box = torch.zeros(pad_size, 6).cuda()
                    input_query_label = padding_label
                    input_query_box = padding_box
                    map_known_indice = torch.tensor([]).to('cuda')
                    if n_boxes:
                        map_known_indice = torch.tensor(range(n_boxes))  
                        map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
                    known_labels_expaned = torch.eye(self.num_classes)[known_labels_expaned].to(bbox_preds.device)
                    input_query_label[map_known_indice] = known_labels_expaned
                    input_query_box[map_known_indice] = known_box_expand

                    dn_q_feature = torch.cat((input_query_box, input_query_label), -1)
                    for i in range(self.pro_enc_repeat):
                        dn_qkv = self.__getattr__(f'qkv{i}')(torch.cat((dn_q_feature, dn_q_feature, dn_q_feature),-1))
                        dn_q, dn_k, dn_v = dn_qkv[:, :self.channel], dn_qkv[:, self.channel:2*self.channel], dn_qkv[:, 2*self.channel:]
                        dn_q_feature = attention(dn_q, dn_k, dn_v, self.nhead, self.channel)
                    dn_q_feature = self.q_linear(dn_q_feature)
                    dn_k_feature = k_feature
                    dn_l2_loss_intra_align = nn.PairwiseDistance(p=2)(dn_q_feature, dn_k_feature.detach())
                    dn_l2_loss_intra = torch.cdist(dn_q_feature, dn_k_feature.detach(), p=2)
                    dn_l2_loss_inter = (torch.cdist(dn_k_feature, dn_k_feature).mean(-1) + 1) ** -1
                    if len(gt_labels) > 1:
                        dn_loss = (dn_l2_loss_intra_align / (dn_l2_loss_intra.sum(-1) - dn_l2_loss_intra_align)) / dn_l2_loss_intra_align.shape[0]
                    else:
                        dn_loss = torch.zeros_like(dn_l2_loss_intra_align)
                    dn_losses += dn_loss/scalars
            else:
                dn_loss = None

            if len(gt_labels) > 1:
                l2_loss_inter = (torch.cdist(k_feature, k_feature, p=2).sum(-1) / (len(gt_labels) - 1)) ** -1
            else:
                l2_loss_inter = torch.zeros_like(torch.cdist(k_feature, k_feature, p=2).mean(-1))
            l2_loss_intra = torch.cdist(q_feature, k_feature, p=2)
            _, min_inds_loss_l2 = l2_loss_intra.min(dim=1)
            _, min_ind_gt = l2_loss_intra.min(dim=0)
            min_inds_loss_l2[min_ind_gt] = torch.arange(0, len(gt_labels), dtype=torch.int64).to(bbox_preds.device)
            min_inds_feature = min_inds_loss_l2

            l2_loss_intra = torch.cdist(q_feature, k_feature.detach(), p=2)
            q_feature_pred = q_feature[min_inds_feature>=0]
            k_feature_pred = k_feature[min_inds_feature[min_inds_feature>=0]]
            l2_loss_intra_align = nn.PairwiseDistance(p=2)(q_feature_pred, k_feature_pred.detach())
            
            if len(gt_labels) >  1:
                mask = mask_matrix[min_inds_feature[min_inds_feature>=0]]
                l2_loss_intra_axis_align = (l2_loss_intra_align / (l2_loss_intra[min_inds_feature>=0].sum(-1)-l2_loss_intra_align)) / l2_loss_intra_align.shape[0]
            else:
                l2_loss_intra_axis_align = torch.zeros_like(l2_loss_intra_align)
            min_inds_feature_raw = torch.ones_like(min_3k_inds) * -1
            min_inds_feature_raw[min_3k_inds >= 0] = min_inds_feature
            min_inds = torch.where(min_3k_inds >= 0, min_inds_feature_raw, -1)

        return min_inds, l2_loss_inter, l2_loss_intra_axis_align, dn_loss