# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/dense_heads/fcaf3d_neck_with_head.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from mmcv.cnn import Scale, bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner.base_module import BaseModule
from torch import nn
from os import path as osp
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models import HEADS, build_loss
from mmdet.core import reduce_mean
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner
from mmdet3d.core.bbox.iou_calculators import axis_aligned_bbox_overlaps_3d
from mmdet3d.models.losses.axis_aligned_iou_loss import axis_aligned_diou
from mmdet3d.core.visualizer.show_result import box_to_corners, _write_obj, _write_oriented_bbox_v2, show_result_v2
from .tr3d_head import varifocal_loss_with_logits

def attention(q, k, v, nhead, channel, deta_p, mask=None):
    q, k, v = [x.view(-1, nhead, channel//nhead).transpose(0,1) for x in (q, k, v)]
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = F.softmax(scores+deta_p, dim=-1)
    q_feature = torch.matmul(attention, v).transpose(0,1).contiguous().view(-1, nhead*d_k)
    return q_feature

def sigmoid_focal_loss(
    logits,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        logits: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as logits. Stores the binary
                 classification label for each element in logits
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

@HEADS.register_module()
class LearnHead(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 out_channels,
                 n_reg_outs,
                 voxel_size,
                 volume_threshold,
                 mode,
                 label2level=None,
                 vfl=False,
                 generative=True,
                 assign_type='volume',
                 top_pts_threshold=6,
                 bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 localization_loss=dict(type='CrossEntropyLoss', use_sigmoid=True),
                 train_cfg=None,
                 test_cfg=None):
        super(LearnHead, self).__init__()
        self.voxel_size = voxel_size
        self.label2level = label2level
        self.volume_threshold = volume_threshold
        self.top_pts_threshold = top_pts_threshold
        self.assign_type = assign_type
        self.generative = generative
        self.vfl = vfl
        # self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.localization_loss = build_loss(localization_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.channel = 256
        self.nhead = 4
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.cost_class = 1.0
        self.cost_giou = 2.0
        self.cost_box = 2.0
        self.mode = mode
        self.dn = True
        self.dn_num = 64
        self.noise_scale = 1.2
        self.num_classes = 18
        self.iter = 0
        self.iter_test = 1
        self.iter_loss = 0
        self.loss_all = {}
        self.temperature = 0.7
        self.show = True
        self.sinkhorn = SinkhornDistance(0.01, 50)
        self._init_layers(in_channels[1:], out_channels, n_reg_outs, n_classes)
        # self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)

    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self._make_up_block(in_channels[i], in_channels[i - 1], generative=self.generative))
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
        self.q_linear = nn.Linear(self.channel, self.channel)
        self.pe_linear = nn.Linear(3, self.nhead)
        self.qkv = nn.Linear(24*3, self.channel*3)
        # self.qkv_box = nn.Linear(6*3, self.channel*3//2)
        # self.qkv_cls = nn.Linear(18*3, self.channel*3//2)
        self.qkv1 = nn.Linear(self.channel*3, self.channel*3)
        self.cross_q = nn.Linear(self.channel, self.channel)
        self.cross_kv = nn.Linear(self.channel*2, self.channel*2)
        self.q_linear1 = nn.Linear(self.channel, self.channel)
        self.pe_linear1 = nn.Linear(3, self.nhead)
        self.k_linear = nn.Linear(24, self.channel)
        # self.k1_linear = nn.Linear(6, self.channel//2)
        # self.k2_linear = nn.Linear(18, self.channel//2)
        self.k_linear1 = nn.Linear(self.channel, self.channel)

    def init_weights(self):
        # for m in self.modules():
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
    def _make_up_block(in_channels, out_channels, generative=False):
        conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
            else ME.MinkowskiConvolutionTranspose
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
                # out_reg = self.__getattr__(f'out_reg_block_{i}')(x)
                # _, cls_pred, localization_pred, point = self._forward_single(out_cls)
                bbox_pred, cls_pred, localization_pred, point = self._forward_single(out_cls)
                # bbox_pred, _, _, _ = self._forward_single(out_reg)
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
        if len(kwargs) > 2:
            loss = self._loss(bbox_preds, cls_preds, localization_preds, points, kwargs["gt_bboxes_3d"][0], kwargs["gt_labels_3d"][0], img_metas)
            # if self.iter_loss == 0:
            #     self.loss_all.update(loss)
            # self.iter_loss += 1
            # for k in loss:
            #     self.loss_all[k] += loss[k]
            # if self.iter_loss % 50 == 0:
            #     for k in loss:
            #         print(f"{self.iter_loss}:, {k}:, {self.loss_all[k]/50}")
            #     for k in loss:
            #         self.loss_all[k] = 0
            # elif self.iter_loss > 311:
            #     self.loss_all = {}
            #     self.iter_loss = 0    
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
                with torch.no_grad():
                    assigned_ids, assigned_ids_box, loss_weight, l2_loss_inter, l2_loss_intra_axis_align, dn_loss = self._get_targets(points, bbox_preds, cls_preds, localization_preds, gt_bboxes, gt_labels, img_meta)
            else:
                with torch.no_grad():
                    assigned_ids = self._get_targets(points, bbox_preds, cls_preds, localization_preds, gt_bboxes, gt_labels, img_meta)
                    assigned_ids_box = assigned_ids
                    l2_loss_inter, l2_loss_intra_axis_align, dn_loss = None, None, None
        else:
            with torch.no_grad():
                assigned_ids = self._get_targets(points, bbox_preds, cls_preds, localization_preds, gt_bboxes, gt_labels, img_meta)
                assigned_ids_box = assigned_ids
                l2_loss_inter, l2_loss_intra_axis_align, dn_loss = None, None, None
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        localization_preds = torch.cat(localization_preds)
        points = torch.cat(points)
        
        # if self.vfl:
        #     cost_weight_raw = torch.where(assigned_ids >= 0, loss_weight, 0)
        # else:
        #     cost_weight_raw = torch.where(assigned_ids >= 0, loss_weight, 0.3)
        # cost_weight_raw = torch.where(assigned_ids >= 0, loss_weight, 1.0)
        # cost_weight_raw = torch.where(assigned_ids >= 0, 1., 0.3)
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0
        # pos_mask_box = assigned_ids_box >= 0
        pos_mask_box = assigned_ids >= 0

        if pos_mask.sum() > 0 and self.mode=="learn":
            cluster_loss_inter = l2_loss_inter 
            cluster_loss_intra = l2_loss_intra_axis_align
        else:
            cluster_loss_inter = None 
            cluster_loss_intra = None
        # loss_weight = loss_weight.detach()
        # cost_weight_raw = cost_weight_raw.detach()
        # pos_loss_weight = cost_weight_raw[pos_mask]
        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        # if self.vfl:
        #     target_label = F.one_hot(cls_targets, n_classes+1)[..., :-1]
        #     target_score = cost_weight_raw.unsqueeze(-1) * target_label
        #     cls_loss = varifocal_loss_with_logits(cls_preds, target_score, target_label)
        # else:
        cls_loss = self.cls_loss(cls_preds, cls_targets)#, weight=cost_weight_raw)
        # bbox loss
        if pos_mask_box.sum() > 0:
            pos_points = points[pos_mask_box]
            pos_bbox_preds = bbox_preds[pos_mask_box]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask_box]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))#, weight=pos_loss_weight)
            l1_loss = F.l1_loss(self._bbox_pred_to_bbox(pos_points, pos_bbox_preds), pos_bbox_targets,
                                reduction='none').sum(-1)# * pos_loss_weight
            #####localization_loss
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
                gt_labels=gt_labels[i])
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
                # l1_loss=torch.sum(torch.cat(l1_losses)) / torch.sum(torch.cat(pos_masks)),
                cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)),
                # localization_loss= 0.1*torch.mean(torch.stack(localization_losses)),
                # cluster_loss_inter = 0.1*torch.sum(torch.cat(cluster_losses_inter)) / len(img_metas),# / torch.sum(torch.cat(pos_masks)),
                # cluster_loss_intra = torch.sum(torch.cat(cluster_losses_intra)) / len(img_metas),
                # dn_loss = torch.sum(torch.cat(dn_losses)) / len(img_metas),
                )
            else:
                return dict(
                bbox_loss=torch.mean(torch.cat(bbox_losses)),
                # l1_loss=torch.sum(torch.cat(l1_losses)) / torch.sum(torch.cat(pos_masks)),
                cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)),
                localization_loss= 0.1*torch.mean(torch.stack(localization_losses)),
                # cluster_loss_inter = 0.1*torch.sum(torch.cat(cluster_losses_inter)) / len(img_metas),# / torch.sum(torch.cat(pos_masks)),
                # cluster_loss_intra = torch.sum(torch.cat(cluster_losses_intra)) / len(img_metas),
                # dn_loss = torch.sum(torch.cat(dn_losses)) / len(img_metas),
                )
        else:
            return dict(
                bbox_loss=torch.mean(torch.cat(bbox_losses)),
                # l1_loss=torch.sum(torch.cat(l1_losses)) / torch.sum(torch.cat(pos_masks)),
                cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)),
                # localization_loss= torch.mean(torch.stack(localization_losses)),
                )

    def _get_bboxes_single(self, bbox_preds, cls_preds, localization_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        # localization_preds = torch.cat(localization_preds).sigmoid()
        theta = 0.5
        # scores = pow(torch.cat(localization_preds).sigmoid(), theta) * pow(torch.cat(cls_preds).sigmoid(), (1 - theta))
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
        bbox_preds = self._bbox_pred_to_bbox(points, bbox_preds)
        center_preds = bbox_preds[:, :3]
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        shape = (n_points, n_boxes, -1)
        points2boxes = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        sample_idx = img_meta["sample_idx"]

        if self.assign_type == 'volume':
            bbox_state = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            label2level = torch.zeros([len(gt_bboxes),])
            downsample_times = [4,3]
            for n in range(len(gt_bboxes)):
                bbox_volume = bbox_state[n][3] * bbox_state[n][4] * bbox_state[n][5]
                for i in range(len(downsample_times)):
                    if bbox_volume > self.volume_threshold * (self.voxel_size * 2 ** downsample_times[i]) ** 3:
                        label2level[n] = 1 - i
                        break                  
        if len(gt_labels) == 0 and self.mode == "learn" and self.iter_test > 0:
            return gt_labels.new_full((n_points,), -1), gt_labels.new_full((n_points,), -1), gt_labels.new_full((n_points,), -1), None, None, None
        elif len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)

        # if self.mode == "tr3d":
        if self.iter_test < 1:
            boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)

            # condition 1: fix level for label
            label2level = gt_labels.new_tensor(self.label2level)
            label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
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
            if self.show:
                self.show_gt_and_pos_box(sample_idx, min_inds, gt_boxes, gt_labels, points, bbox_preds, cls_preds)

            return min_inds
        elif self.mode == "sinkhorn":
            gt_cls_onehot = F.one_hot(gt_labels, self.num_classes).float()
            mat_class = sigmoid_focal_loss(unsigmoid_cls_pred.unsqueeze(1).expand(shape), gt_cls_onehot.unsqueeze(0).expand(shape), self.focal_loss_alpha, self.focal_loss_gamma).sum(-1)
            mat_bg_class = sigmoid_focal_loss(unsigmoid_cls_pred, torch.zeros_like(unsigmoid_cls_pred), self.focal_loss_alpha, self.focal_loss_gamma).sum(-1)
            boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            # enlarged_boxes = gt_bboxes.enlarged_box(0.2)
            # is_in_boxes = enlarged_boxes.points_in_boxes_all(points) # N M
            is_in_boxes = gt_bboxes.points_in_boxes_all(points) # N M
            gt_boxes = boxes[:, :6].to(points.device)
            pred_boxes = bbox_preds
            mat_box = torch.cdist(pred_boxes, gt_boxes, p=1)
            pred_xyzxyz = torch.cat((pred_boxes[:, :3] - pred_boxes[:, 3:] / 2, pred_boxes[:, :3] + pred_boxes[:, 3:] / 2), dim=-1)
            gt_xyzxyz = torch.cat((gt_boxes[:, :3] - gt_boxes[:, 3:] / 2, gt_boxes[:, :3] + gt_boxes[:, 3:] / 2), dim=-1)
            mat_giou = -axis_aligned_bbox_overlaps_3d(pred_xyzxyz, gt_xyzxyz, mode="giou")
            iou = axis_aligned_bbox_overlaps_3d(pred_xyzxyz, gt_xyzxyz, mode="iou")
            cost = self.cost_class * mat_class + self.cost_giou * mat_giou + float_max*(1 - is_in_boxes.float()) #+ self.cost_box * mat_box
            label2level = gt_labels.new_tensor(self.label2level)
            label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
            point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
            level_condition = label_levels == point_levels
            cost = torch.where(level_condition, cost, float_max)
            topk_ious, _ = torch.topk(iou*is_in_boxes.float(), self.top_pts_threshold, dim=0)
            mu = iou.new_ones(n_boxes + 1) * self.top_pts_threshold
            # mu[:-1] = torch.clamp(topk_ious.sum(0).int(), min=1).float()
            mu[-1] = n_points - mu[:-1].sum()
            nu = iou.new_ones(n_points)
            cost = torch.cat([cost, mat_bg_class.unsqueeze(1)], dim=1)
            print(cost.shape, mu.sum())
            _, pi = self.sinkhorn(mu, nu, cost.T)
            rescale_factor, _ = pi.max(dim=1)
            pi = pi / rescale_factor.unsqueeze(1)
            max_assigned_units, matched_gt_inds = torch.max(pi, dim=0)

            for i in range(n_boxes):
                if len(matched_gt_inds[matched_gt_inds==i]) > self.top_pts_threshold:
                    mask = (matched_gt_inds==i)
                    topk_units, topk_ind = torch.topk(max_assigned_units[mask], self.top_pts_threshold, dim=0)
                    matched_gt_inds[mask] = n_boxes
                    temp = matched_gt_inds[mask]
                    temp[topk_ind] = i
                    matched_gt_inds[mask] = temp
            matched_gt_inds[matched_gt_inds==matched_gt_inds.max()] = -1
            min_inds = matched_gt_inds
            if self.show:
                self.show_gt_and_pos_box(sample_idx, min_inds, gt_boxes, gt_labels, points, pred_boxes, cls_preds)
            return min_inds
        elif self.mode == "cost":
            gt_cls_onehot = F.one_hot(gt_labels, self.num_classes).float()
            mat_class = sigmoid_focal_loss(unsigmoid_cls_pred.unsqueeze(1).expand(shape), gt_cls_onehot.unsqueeze(0).expand(shape),
                                        self.focal_loss_alpha, self.focal_loss_gamma).sum(-1)
            boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            gt_boxes = boxes[:, :6].to(points.device)
            is_in_boxes = gt_bboxes.points_in_boxes_all(points) # N M
            pred_boxes = bbox_preds
            mat_box = torch.cdist(pred_boxes, gt_boxes, p=1)
            pred_xyzxyz = torch.cat((pred_boxes[:, :3] - pred_boxes[:, 3:] / 2, pred_boxes[:, :3] + pred_boxes[:, 3:] / 2), dim=-1)
            gt_xyzxyz = torch.cat((gt_boxes[:, :3] - gt_boxes[:, 3:] / 2, gt_boxes[:, :3] + gt_boxes[:, 3:] / 2), dim=-1)
            mat_giou = -axis_aligned_bbox_overlaps_3d(pred_xyzxyz, gt_xyzxyz, mode="giou")
            iou = axis_aligned_bbox_overlaps_3d(pred_xyzxyz, gt_xyzxyz, mode="iou")
            
            # condition 1: fix level for label
            label2level = gt_labels.new_tensor(self.label2level)
            label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
            point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
            level_condition = label_levels == point_levels

            # condition 2: keep topk location per box by center distance
            center = boxes[..., :3]
            cost = self.cost_class * mat_class + self.cost_giou * mat_giou + float_max*(1 - is_in_boxes.float())
            cost = torch.where(level_condition, cost, float_max)
            topk_distances = torch.topk(cost,
                                        min(self.top_pts_threshold + 1, len(cost)),
                                        largest=False, dim=0).values[-1]
            topk_condition = cost < topk_distances.unsqueeze(0)

            # condition 3.0: only closest object to point
            cost = self.cost_class * mat_class + self.cost_giou * mat_giou + float_max*(1 - is_in_boxes.float())
            _, min_inds_ = cost.min(dim=1)

            # condition 3: min center distance to box per point
            cost = torch.where(topk_condition, cost, float_max)
            min_values, min_ids = cost.min(dim=1)
            min_inds = torch.where(min_values < float_max, min_ids, -1)
            min_inds = torch.where(min_inds == min_inds_, min_ids, -1)
            if self.show:
                self.show_gt_and_pos_box(sample_idx, min_inds, gt_boxes, gt_labels, points, pred_boxes, cls_preds)

            return min_inds
        # elif self.mode == "learn":
        else:
            boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            gt_boxes = boxes[:, :6].to(points.device)
            boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
            pred_xyzxyz = torch.cat((bbox_preds[:, :3] - bbox_preds[:, 3:] / 2, bbox_preds[:, :3] + bbox_preds[:, 3:] / 2), dim=-1)
            gt_xyzxyz = torch.cat((gt_boxes[:, :3] - gt_boxes[:, 3:] / 2, gt_boxes[:, :3] + gt_boxes[:, 3:] / 2), dim=-1)
            mask_matrix = gt_labels[:, None] != gt_labels
            # condition 1: fix level for label
            label2level = gt_labels.new_tensor(self.label2level)
            label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
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
                
            selected_box_feature = torch.cat((selected_bbox_preds, selected_pred_cls), -1)
            gt_cls = torch.eye(18)[gt_labels].to(bbox_preds.device)
            # box_feature = self.qkv_box(torch.cat((selected_bbox_preds, selected_bbox_preds, selected_bbox_preds), -1))
            # cls_feature = self.qkv_cls(torch.cat((selected_pred_cls, selected_pred_cls, selected_pred_cls), -1))
            # qkv_feature = torch.cat((box_feature, cls_feature), -1)
            gt_feature = torch.cat((gt_boxes, gt_cls), -1)
            qkv = self.qkv(torch.cat((selected_box_feature, selected_box_feature, selected_box_feature),-1))
            # qkv = self.qkv1(qkv_feature)
            q, k, v = qkv[:, :self.channel], qkv[:, self.channel:2*self.channel], qkv[:, 2*self.channel:]
            corners = box_to_corners(selected_bbox_preds)
            deta_p = 0
            for i in range(corners.shape[-1]):
                deta_p += self.pe_linear(torch.tanh(torch.einsum('nc, mc->nmc', selected_points, corners[:,i])))
            q_feature = attention(q, k, v, self.nhead, self.channel, deta_p.permute(2,0,1))
            q_feature += q
            q_feature = self.q_linear(q_feature)
            qkv1 = self.qkv1(torch.cat((q_feature, q_feature, q_feature),-1))
            q1, k1, v1 = qkv1[:, :self.channel], qkv1[:, self.channel:2*self.channel], qkv1[:, 2*self.channel:]
            deta_p = 0
            for i in range(corners.shape[-1]):
                deta_p += self.pe_linear1(torch.tanh(torch.einsum('nc, mc->nmc', selected_points, corners[:,i])))
            q_feature1 = attention(q1, k1, v1, self.nhead, self.channel, deta_p.permute(2,0,1))
            q_feature1 += q1
            q_feature = self.q_linear1(q_feature1)

            # box_feature = self.k1_linear(gt_boxes)
            # cls_feature = self.k2_linear(gt_cls)
            # k_feature = self.k_linear(torch.cat((box_feature, cls_feature), -1))
            k_feature = self.k_linear(gt_feature)

            if sample_idx == 'scene0092_03':
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, perplexity=k_feature.shape[0]-1, init='pca', random_state=0)
                result = tsne.fit_transform(k_feature.detach().cpu().numpy())
                fig = self.plot_embedding(result, range(len(result)),
                         't-SNE embedding of the gt_feature')
                out_dir = '/data/lxr/visual/assign_scannet_gt_tsne'

                mmcv.mkdir_or_exist(out_dir)
                plt.savefig(f'/data/lxr/visual/assign_scannet_gt_tsne/gt_feature_{self.iter}.jpg')
                plt.show()
                tsne = TSNE(n_components=2, perplexity=q_feature.shape[0]-1, init='pca', random_state=0)
                result = tsne.fit_transform(q_feature.detach().cpu().numpy())
                fig = self.plot_embedding(result, min_3k_inds[min_3k_inds>=0].detach().cpu().numpy(),
                         't-SNE embedding of the pred_feature')
                out_dir = '/data/lxr/visual/assign_scannet_pred_tsne'

                mmcv.mkdir_or_exist(out_dir)
                plt.savefig(f'/data/lxr/visual/assign_scannet_pred_tsne/pred_feature_{self.iter}.jpg')
                plt.show()
                plt.close()
            

            if self.dn:
                 scalars = 1
                 dn_losses = torch.zeros_like(nn.PairwiseDistance(p=2)(k_feature, k_feature.detach()))
                 for i in range(scalars):
                    scalar = 1 # test for scalar set 1, more need set pad and mask
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
                        map_known_indice = torch.tensor(range(n_boxes))  # [1,2, 1,2,3]
                        map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
                    known_labels_expaned = torch.eye(self.num_classes)[known_labels_expaned].to(bbox_preds.device)
                    input_query_label[map_known_indice] = known_labels_expaned
                    input_query_box[map_known_indice] = known_box_expand

                    input_query = torch.cat((input_query_box, input_query_label), -1)
                    dn_qkv = self.qkv(torch.cat((input_query, input_query, input_query),-1))
                    # dn_box_feature = self.qkv_box(torch.cat((input_query_box, input_query_box, input_query_box),-1))
                    # dn_cls_feature = self.qkv_cls(torch.cat((input_query_label, input_query_label, input_query_label),-1))
                    # dn_qkv = self.qkv1(torch.cat((dn_box_feature, dn_cls_feature), -1))
                    dn_q, dn_k, dn_v = dn_qkv[:, :self.channel], dn_qkv[:, self.channel:2*self.channel], dn_qkv[:, 2*self.channel:]
                    dn_corners = box_to_corners(input_query_box)
                    dn_points = input_query_box[:, :3]
                    dn_deta_p = 0
                    for i in range(dn_corners.shape[-1]):
                        dn_deta_p += self.pe_linear(torch.tanh(torch.einsum('nc, mc->nmc', dn_points, dn_corners[:,i])))
                    dn_q_feature = attention(dn_q, dn_k, dn_v, self.nhead, self.channel, dn_deta_p.permute(2,0,1))
                    # dn_attention = torch.einsum("hnc, hmc->hnm", dn_q, dn_k)
                    # dn_q_feature = torch.einsum("nm, mc->nc", F.softmax(dn_attention / self.channel**0.5, dim=1), dn_v)
                    dn_q_feature += dn_q
                    dn_q_feature = self.q_linear(dn_q_feature)
                    qkv1 = self.qkv1(torch.cat((dn_q_feature, dn_q_feature, dn_q_feature),-1))
                    q1, k1, v1 = qkv1[:, :self.channel], qkv1[:, self.channel:2*self.channel], qkv1[:, 2*self.channel:]
                    dn_deta_p = 0
                    for i in range(dn_corners.shape[-1]):
                        dn_deta_p += self.pe_linear1(torch.tanh(torch.einsum('nc, mc->nmc', dn_points, dn_corners[:,i])))
                    q_feature1 = attention(q1, k1, v1, self.nhead, self.channel, dn_deta_p.permute(2,0,1))
                    q_feature1 += q1
                    dn_q_feature = self.q_linear1(q_feature1)

                    # dn_q1 = self.cross_q(k_feature)
                    # dn_kv1 = self.cross_kv(torch.cat((dn_q_feature, dn_q_feature),-1))
                    # dn_k1, dn_v1 = dn_kv1[:, :self.channel], dn_kv1[:, self.channel:2*self.channel]
                    # dn_k_feature1 = attention(dn_q1, dn_k1, dn_v1, self.nhead, self.channel)
                    # dn_k_feature1 += dn_q1
                    # dn_k_feature = self.q_linear1(dn_k_feature1)
                    dn_k_feature = k_feature
                    dn_l2_loss_intra_align = nn.PairwiseDistance(p=2)(dn_q_feature, dn_k_feature.detach())
                    dn_l2_loss_intra = torch.cdist(dn_q_feature, dn_k_feature.detach(), p=2)
                    dn_l2_loss_inter = (torch.cdist(dn_k_feature, dn_k_feature).mean(-1) + 1) ** -1
                    # dn_loss = (0.5 * dn_l2_loss_intra_align / dn_l2_loss_intra.sum(-1)) / dn_l2_loss_intra_align.shape[0]
                    # dn_sim_intra = torch.cosine_similarity(dn_q_feature.unsqueeze(1), dn_k_feature.detach().unsqueeze(0), dim=2)
                    # dn_sim_intra_align = torch.cosine_similarity(dn_q_feature, dn_k_feature.detach())
                    if len(gt_labels) > 1:
                        dn_loss = (dn_l2_loss_intra_align / (dn_l2_loss_intra.sum(-1) - dn_l2_loss_intra_align)) / dn_l2_loss_intra_align.shape[0]
                        # dn_loss = (dn_l2_loss_intra_align / ((dn_l2_loss_intra*mask_matrix).sum(-1) + 1) * mask_matrix.sum(-1)) / dn_l2_loss_intra_align.shape[0]
                        # dn_loss = -torch.log(torch.exp(dn_sim_intra_align / self.temperature) / ((torch.exp(dn_sim_intra / self.temperature)).sum(-1) - torch.exp(dn_sim_intra_align / self.temperature))) / dn_sim_intra_align.shape[0]
                    else:
                        dn_loss = torch.zeros_like(dn_l2_loss_intra_align)
                    dn_losses += dn_loss/scalars
                # dn_loss = -torch.log(torch.exp(dn_l2_loss_intra_align / self.temperature) / torch.exp(dn_l2_loss_intra / self.temperature).sum(-1)) / dn_l2_loss_intra_align.shape[0]
            # q1 = self.cross_q(k_feature)
            # kv1 = self.cross_kv(torch.cat((q_feature, q_feature),-1))
            # k1, v1 = kv1[:, :self.channel], kv1[:, self.channel:2*self.channel]
            # q_feature1 = attention(q1, k1, v1, self.nhead, self.channel)
            # q_feature1 += q1
            # k_feature = self.q_linear1(q_feature1)
            if len(gt_labels) > 1:
                l2_loss_inter = (torch.cdist(k_feature, k_feature, p=2).sum(-1) / (len(gt_labels) - 1)) ** -1 #  / len(gt_labels)
                # l2_loss_inter = ((torch.cdist(k_feature, k_feature, p=2)*mask_matrix).sum(-1) + 1) ** -1 * mask_matrix.sum(-1) #  / len(gt_labels)
            else:
                l2_loss_inter = torch.zeros_like(torch.cdist(k_feature, k_feature, p=2).mean(-1))
            # l2_loss_inter = ((torch.cosine_similarity(k_feature.unsqueeze(1), k_feature.unsqueeze(0), dim=2) + 1).sum(-1) - 2)  / (len(gt_labels)*len(gt_labels))
            # if self.dn:
            #     l2_loss_inter += dn_l2_loss_inter
            l2_loss_intra = torch.cdist(q_feature, k_feature, p=2)
            # l2_loss_intra = torch.where(selected_level_condition, l2_loss_intra, float_max)
            # topk_l2 = torch.topk(l2_loss_intra, min(self.top_pts_threshold + 1, len(l2_loss_intra)), largest=False, dim=0).values[-1]
            # topk_condition = l2_loss_intra < topk_l2.unsqueeze(0)
            # l2_loss_intra = torch.cdist(q_feature, k_feature, p=2)
            # _, min_inds_ = l2_loss_intra.min(dim=1)
            # l2_loss_intra = torch.where(topk_condition, l2_loss_intra, float_max)
            # min_values, min_ids = l2_loss_intra.min(dim=1)
            # min_inds_loss_l2 = torch.where(min_values < float_max, min_ids, -1)
            # min_inds_loss_l2 = torch.where(min_inds_loss_l2 == min_inds_, min_ids, -1)
            _, min_inds_loss_l2 = l2_loss_intra.min(dim=1)
            _, min_ind_gt = l2_loss_intra.min(dim=0)
            # min_ind_cls = torch.ones_like(min_inds_loss_l2) * -1
            # min_ind_cls[min_ind_gt] = torch.arange(0, len(gt_labels), dtype=torch.int64).to(bbox_preds.device)
            min_inds_loss_l2[min_ind_gt] = torch.arange(0, len(gt_labels), dtype=torch.int64).to(bbox_preds.device)
            # l2_loss_intra[min_ind_gt] = float_max
            # min_inds_feature = min_inds_loss_l2
            min_inds_feature = min_3k_inds[min_3k_inds>=0]

            # iou weight
            gt_l2_align_xyzxyz = gt_xyzxyz[min_inds_feature[min_inds_feature>=0]]
            selected_boxes = selected_bbox_preds[min_inds_feature>=0]
            selected_pred_xyzxyz = torch.cat((selected_boxes[:, :3] - selected_boxes[:, 3:] / 2, selected_boxes[:, :3] + selected_boxes[:, 3:] / 2), dim=-1)
            iou_weight = axis_aligned_bbox_overlaps_3d(selected_pred_xyzxyz, gt_l2_align_xyzxyz, mode="iou", is_aligned=True)
            iou_weight_raw = torch.zeros_like(min_inds_feature).float()
            iou_weight = torch.where(iou_weight > 0.1, iou_weight, 0.1)
            iou_weight_raw[min_inds_feature >= 0] = iou_weight
            iou_weight_raw[min_ind_gt] = 1.
            l2_loss_intra = torch.cdist(q_feature, k_feature.detach(), p=2)
            q_feature_pred = q_feature[min_inds_feature>=0]
            # k_feature_pred = k_feature[min_inds_loss_l2]
            k_feature_pred = k_feature[min_inds_feature[min_inds_feature>=0]]
            l2_loss_intra_align = nn.PairwiseDistance(p=2)(q_feature_pred, k_feature_pred.detach())
            # l2_loss_intra_axis_align = (0.5 * l2_loss_intra_align / l2_loss_intra[min_inds_feature>=0].sum(-1)) / l2_loss_intra_align.shape[0]
            # sim_intra = torch.cosine_similarity(q_feature.unsqueeze(1), k_feature.detach().unsqueeze(0), dim=2)
            # sim_intra_align = torch.cosine_similarity(q_feature_pred, k_feature_pred.detach())
            if len(gt_labels) >  1:
                mask = mask_matrix[min_inds_feature[min_inds_feature>=0]]
                l2_loss_intra_axis_align = (l2_loss_intra_align / (l2_loss_intra[min_inds_feature>=0].sum(-1)-l2_loss_intra_align)) / l2_loss_intra_align.shape[0]
                # l2_loss_intra_axis_align = (l2_loss_intra_align / ((l2_loss_intra[min_inds_feature>=0]*mask).sum(-1)+1) * mask.sum(-1)) / l2_loss_intra_align.shape[0]
                # l2_loss_intra_axis_align = -torch.log(torch.exp(sim_intra_align / self.temperature) / ((torch.exp(sim_intra[min_inds_feature>=0] / self.temperature)).sum(-1) - torch.exp(sim_intra_align / self.temperature))) / sim_intra_align.shape[0]
            else:
                l2_loss_intra_axis_align = torch.zeros_like(l2_loss_intra_align)
            min_inds_feature_raw = torch.ones_like(min_3k_inds) * -1
            min_inds_feature_raw[min_3k_inds >= 0] = min_inds_feature
            min_inds = torch.where(min_3k_inds >= 0, min_inds_feature_raw, -1)
            cost_weight_raw = torch.zeros_like(min_inds).float()
            cost_weight_raw[min_3k_inds >= 0] = iou_weight_raw

        
        if self.show:
           self.show_gt_and_pos_box(sample_idx, min_inds, gt_boxes, gt_labels, points, bbox_preds, cls_preds)

        if self.dn:
            return min_inds, min_3k_inds, cost_weight_raw, l2_loss_inter, l2_loss_intra_axis_align, dn_loss
        else:
            return min_inds, min_3k_inds, cost_weight_raw, l2_loss_inter, l2_loss_intra_axis_align, None

    def show_gt_and_pos_box(self, sample_idx, min_inds, gt_boxes, gt_labels, points, pred_boxes, cls_preds):
        if sample_idx == 'scene0092_03':
            out_dir = '/data/lxr/visual/assign/scene0092_03'
            filename = 'scene0092_03'
            result_path = osp.join(out_dir, self.mode, filename)
            mmcv.mkdir_or_exist(result_path)
            colors = np.multiply([plt.cm.get_cmap('nipy_spectral', 19)((i * 5 + 11) % 18 + 1)[:3] for i in range(18)], 255).astype(np.uint8)
            # pos_mask = min_inds>=0
            pos_mask = min_inds>=0
            # _write_obj(np.concatenate((points[pos_mask].cpu().numpy(), colors[min_inds[pos_mask].cpu().numpy()]), -1), f'{result_path}_{self.iter}_assign_points.obj')
            _write_obj(np.concatenate((points[pos_mask].cpu().numpy(), colors[min_inds[pos_mask].cpu().numpy()]), -1), f'{result_path}_{self.iter}_assign_points.obj')
            selected_pred_boxes = pred_boxes[pos_mask]
            selected_pred_clses = cls_preds[pos_mask]
            selected_pred_clses = selected_pred_clses.max(-1)[1].detach().cpu().numpy()
            selected_pred_coners = box_to_corners(selected_pred_boxes).detach().cpu().numpy()
            _write_oriented_bbox_v2(selected_pred_coners, selected_pred_clses, f'{result_path}_{self.iter}_assign_pred.obj')
            # _write_oriented_bbox_v2(selected_pred_coners, min_inds[pos_mask].cpu().numpy(), f'{result_path}_{self.iter}_assign_pred.obj')
            gt_corners = box_to_corners(gt_boxes).detach().cpu().numpy()
            _write_oriented_bbox_v2(gt_corners,  gt_labels.detach().cpu().numpy(), f'{result_path}_{self.iter}_assign_gt.obj')
            self.iter += 1
        if sample_idx == 'scene0011_00':
            out_dir = '/data/lxr/visual/assign_test/scene0011_00'
            filename = 'scene0011_00'
            result_path = osp.join(out_dir, self.mode, filename)
            mmcv.mkdir_or_exist(result_path)
            colors = np.multiply([plt.cm.get_cmap('nipy_spectral', 50)((i * 5 + 11) % 49 + 1)[:3] for i in range(50)], 255).astype(np.uint8)
            pos_mask = min_inds>=0
            _write_obj(np.concatenate((points[pos_mask].cpu().numpy(), colors[min_inds[pos_mask].cpu().numpy()]), -1), f'{result_path}_{self.iter_test}_assign_points.obj')
            selected_pred_boxes = pred_boxes[pos_mask]
            selected_pred_clses = cls_preds[pos_mask]
            selected_pred_clses = selected_pred_clses.max(-1)[1].detach().cpu().numpy()
            selected_pred_coners = box_to_corners(selected_pred_boxes).detach().cpu().numpy()
            _write_oriented_bbox_v2(selected_pred_coners, selected_pred_clses, f'{result_path}_{self.iter_test}_assign_pred.obj')
            # _write_oriented_bbox_v2(selected_pred_coners, min_inds[pos_mask].cpu().numpy(), f'{result_path}_{self.iter}_assign_pred.obj')
            gt_corners = box_to_corners(gt_boxes).detach().cpu().numpy()
            _write_oriented_bbox_v2(gt_corners,  gt_labels.detach().cpu().numpy(), f'{result_path}_{self.iter_test}_assign_gt.obj')
            self.iter_test += 1


    def plot_embedding(self, data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                    color=plt.cm.Set1(label[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        return fig