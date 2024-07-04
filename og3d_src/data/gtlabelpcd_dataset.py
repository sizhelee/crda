import os
import jsonlines
import json
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from .common import pad_tensors, gen_seq_masks
    from .gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES
except:
    from common import pad_tensors, gen_seq_masks
    from gtlabel_dataset import GTLabelDataset, ROTATE_ANGLES
from copy import deepcopy

class GTLabelPcdDataset(GTLabelDataset):
    def __init__(
        self, scan_id_file, anno_file, scan_dir, category_file,
        cat2vec_file=None, keep_background=False, random_rotate=False,
        num_points=1024, max_txt_len=50, max_obj_len=80,
        in_memory=False, gt_scan_dir=None, iou_replace_gt=0, is_train=False, cfg=None, 
    ):
        super().__init__(
            scan_id_file, anno_file, scan_dir, category_file,
            cat2vec_file=cat2vec_file, keep_background=keep_background,
            random_rotate=random_rotate, 
            max_txt_len=max_txt_len, max_obj_len=max_obj_len,
            gt_scan_dir=gt_scan_dir, iou_replace_gt=iou_replace_gt, 
            cfg=cfg, 
        )
        self.num_points = num_points
        self.in_memory = in_memory

        if self.in_memory:
            for scan_id in self.scan_ids:
                self.get_scan_pcd_data(scan_id)
                if self.gt_scan_dir:
                    self.get_scan_gt_pcd_data(scan_id)

        with open("./nr3d_vil3dref_10_val.json", "r", encoding='utf-8') as f:
            self.change_info = json.load(f)
        self.idx_ls = []
        if is_train:
            self.split = "train"
        else:
            self.split = "val"

        with open("./obj_feats_all.json", "r") as f:
            self.obj_feats_all = json.load(f)

    def get_scan_pcd_data(self, scan_id):
        if self.in_memory and 'pcds' in self.scans[scan_id]:
            return self.scans[scan_id]['pcds']
        
        pcd_data = torch.load(
            os.path.join(self.scan_dir, 'pcd_with_global_alignment_pred', '%s.pth'%scan_id)
        )
        points, colors, ious = pcd_data[0], pcd_data[1], pcd_data[2]
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)
        instance_labels = pcd_data[-1]
        obj_pcds = []
        for i in range(instance_labels.max() + 1):
            mask = instance_labels == i     # time consuming
            if mask.sum() == 0:
                obj_pcds.append(np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float32))
            else:
                obj_pcds.append(pcds[mask])
        if self.in_memory:
            self.scans[scan_id]['pcds'] = obj_pcds
            self.scans[scan_id]['ious'] = ious
        return obj_pcds

    def get_scan_gt_pcd_data(self, scan_id):
        if self.in_memory and 'gt_pcds' in self.scans[scan_id]:
            return self.scans[scan_id]['gt_pcds']
        
        pcd_data = torch.load(
            os.path.join(self.gt_scan_dir, 'pcd_with_global_alignment', '%s.pth'%scan_id)
        )
        points, colors = pcd_data[0], pcd_data[1]
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)
        instance_labels = pcd_data[-1]
        obj_pcds = []
        for i in range(instance_labels.max() + 1):
            mask = instance_labels == i     # time consuming
            obj_pcds.append(pcds[mask])
        if self.in_memory:
            self.scans[scan_id]['gt_pcds'] = obj_pcds
        return obj_pcds

    def _get_obj_inputs(self, obj_pcds, obj_colors, obj_labels, obj_ids, tgt_obj_idx, obj_ious, obj_pcd_gt, theta=None):
        tgt_obj_type = obj_labels[tgt_obj_idx]
        if (self.max_obj_len is not None) and (len(obj_labels) > self.max_obj_len):
            selected_obj_idxs = [tgt_obj_idx]
            remained_obj_idxs = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj != tgt_obj_idx:
                    if klabel == tgt_obj_type:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idxs.append(kobj)
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idxs)
                selected_obj_idxs += remained_obj_idxs[:self.max_obj_len - len(selected_obj_idxs)]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_colors = [obj_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]
            obj_ious = [obj_ious[i] for i in selected_obj_idxs]
            tgt_obj_idx = 0

        if (theta is not None) and (theta != 0):
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            rot_matrix = None

        obj_fts, obj_locs = [], []
        for obj_pcd in obj_pcds:
            # obj_center = obj_pcd[:, :3].mean(0)
            # obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            # obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            try:
                obj_center = obj_pcd[:, :3].mean(0)
                obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            except:
                obj_center = np.array([0, 0, 0])
                obj_size = np.array([0, 0, 0])
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            # obj_locs[-1][:3] = obj_center
            # sample points
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points, replace=(len(obj_pcd) < self.num_points))
            obj_pcd = obj_pcd[pcd_idxs]
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        if rot_matrix is not None:
            obj_pcd_gt[:, :3] = np.matmul(obj_pcd_gt[:, :3], rot_matrix.transpose())
        obj_center = obj_pcd_gt[:, :3].mean(0)
        obj_size = obj_pcd_gt[:, :3].max(0) - obj_pcd_gt[:, :3].min(0)
        obj_loc_gt = np.concatenate([obj_center, obj_size], 0)

        obj_fts = np.stack(obj_fts, 0)
        obj_locs = np.array(obj_locs)
        obj_colors = np.array(obj_colors)
        obj_ious = np.array(obj_ious)
            
        return obj_fts, obj_locs, obj_colors, obj_labels, obj_ids, tgt_obj_idx, obj_ious, obj_loc_gt

    def __getitem__(self, idx):
        item = self.data[idx]
        item_id = item['item_id']
        scan_id = item['scan_id']
        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        tgt_obj_idx = item['target_id']
        tgt_obj_type = item['instance_type']

        txt_tokens = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        txt_lens = len(txt_tokens)

        if 'enc_tokens_mask' in item.keys():
            txt_tokens_mask = torch.LongTensor(item['enc_tokens_mask'][:self.max_txt_len])
        else:
            txt_tokens_mask = torch.LongTensor(item['enc_tokens'][:self.max_txt_len])
        txt_lens_mask = len(txt_tokens_mask)

        # if self.gt_scan_dir is None or item['max_iou'] > self.iou_replace_gt:
        if True:
            obj_pcds = self.get_scan_pcd_data(scan_id)
            obj_pcd_gt = self.get_scan_gt_pcd_data(scan_id)[tgt_obj_idx]
            obj_labels = self.scans[scan_id]['inst_labels']
            obj_gmm_colors = self.scans[scan_id]['inst_colors']

            gt_inst_locs = self.scans[scan_id]['gt_inst_locs'][tgt_obj_idx]
            max_iou, tgt_obj_idx = 0, 0
            obj_ious = []
            pred_inst_locs = self.scans[scan_id]['inst_locs']
            for i in range(pred_inst_locs.shape[0]):
                now_iou = calc_3diou(gt_inst_locs, pred_inst_locs[i])
                obj_ious.append(now_iou)
                if now_iou > max_iou:
                    max_iou = now_iou
                    tgt_obj_idx = i

        else:
            tgt_obj_idx = item['gt_target_id']
            obj_pcds = self.get_scan_gt_pcd_data(scan_id)
            obj_labels = self.scans[scan_id]['gt_inst_labels']
            obj_gmm_colors = self.scans[scan_id]['gt_inst_colors']
        obj_ids = [str(x) for x in range(len(obj_labels))]

        if not self.keep_background:
            selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if obj_label not in ['wall', 'floor', 'ceiling']]
            tgt_obj_idx = selected_obj_idxs.index(tgt_obj_idx)
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_gmm_colors = [obj_gmm_colors[i] for i in selected_obj_idxs]
            obj_ids = [obj_ids[i] for i in selected_obj_idxs]
            obj_ious = [obj_ious[i] for i in selected_obj_idxs]

        if self.random_rotate:
            theta_idx = np.random.randint(len(ROTATE_ANGLES))
            theta = ROTATE_ANGLES[theta_idx]
        else:
            theta = 0

        change_obj = 0
        if self.split == "train":
            if idx in self.idx_ls:
                change_obj = 1
            else:
                self.idx_ls.append(idx)

        if change_obj:
            if not tgt_obj_type in self.obj_feats_all.keys():
                change_obj = 0
                new_obj_fts = np.zeros(768)
            else:
                feats_ls = random.sample(self.obj_feats_all[tgt_obj_type], min(10, len(self.obj_feats_all[tgt_obj_type])))
                feats_ls = np.array(feats_ls).mean(0)
                new_obj_fts = feats_ls
        else:
            if "feats" in item.keys():
                new_obj_fts = item["feats"]
            else:
                new_obj_fts = None

        # if change_obj:
        #     able_change = 1
        #     for item_item in self.change_info:
        #         item_id_now = item_item["item_id"]
        #         if item_id_now != item_id:
        #             continue
        #         if len(item_item["change_list"]) == 0:
        #             able_change = 0
        #             break
        #         scan_id_new, tgt_idx_new = random.choice(item_item["change_list"])

        #         if scan_id_new != scan_id:

        #             if self.val_dataset.gt_scan_dir is None or item_item['max_iou'] > self.val_dataset.iou_replace_gt:
        #                 obj_pcds_new = self.val_dataset.get_scan_pcd_data(scan_id_new)
        #                 obj_gmm_colors_new = self.val_dataset.scans[scan_id_new]['inst_colors']
        #                 obj_labels_new = self.val_dataset.scans[scan_id_new]['inst_labels']
        #             else:
        #                 tgt_idx_new = item_item['gt_target_id']
        #                 obj_pcds_new = self.val_dataset.get_scan_gt_pcd_data(scan_id_new)
        #                 obj_gmm_colors_new = self.val_dataset.scans[scan_id_new]['gt_inst_colors']
        #                 obj_labels_new = self.val_dataset.scans[scan_id_new]['gt_inst_labels']
                
        #         else:
        #             if self.gt_scan_dir is None or item_item['max_iou'] > self.iou_replace_gt:
        #                 obj_pcds_new = self.get_scan_pcd_data(scan_id_new)
        #                 obj_gmm_colors_new = self.scans[scan_id_new]['inst_colors']
        #                 obj_labels_new = self.scans[scan_id_new]['inst_labels']
        #             else:
        #                 tgt_idx_new = item_item['gt_target_id']
        #                 obj_pcds_new = self.get_scan_gt_pcd_data(scan_id_new)
        #                 obj_gmm_colors_new = self.scans[scan_id_new]['gt_inst_colors']
        #                 obj_labels_new = self.scans[scan_id_new]['gt_inst_labels']


        #         obj_pcds_safe = deepcopy(obj_pcds)
        #         obj_pcds_safe[tgt_obj_idx] = change_xyz(obj_pcds_safe[tgt_obj_idx], obj_pcds_new[tgt_idx_new])
        #         obj_gmm_colors_safe = deepcopy(obj_gmm_colors)
        #         obj_gmm_colors_safe[tgt_obj_idx] = obj_gmm_colors_new[tgt_idx_new]
        #         obj_labels_safe = deepcopy(obj_labels)
        #         obj_labels_safe[tgt_obj_idx] = obj_labels_new[tgt_idx_new]

        #         aug_obj_fts, aug_obj_locs, aug_obj_gmm_colors, aug_obj_labels, \
        #             aug_obj_ids, aug_tgt_obj_idx = self._get_obj_inputs(
        #                 obj_pcds_safe, obj_gmm_colors_safe, obj_labels_safe, obj_ids, tgt_obj_idx,
        #                 theta=theta
        #             )
        #         break

        #     if able_change == 0:
        #         aug_obj_fts, aug_obj_locs, aug_obj_gmm_colors, aug_obj_labels, \
        #             aug_obj_ids, aug_tgt_obj_idx = self._get_obj_inputs(
        #                 obj_pcds, obj_gmm_colors, obj_labels, obj_ids, tgt_obj_idx,
        #                 theta=theta
        #             )

        # else:
        #     aug_obj_fts, aug_obj_locs, aug_obj_gmm_colors, aug_obj_labels, \
        #         aug_obj_ids, aug_tgt_obj_idx = self._get_obj_inputs(
        #             obj_pcds, obj_gmm_colors, obj_labels, obj_ids, tgt_obj_idx,
        #             theta=theta
        #         )

        aug_obj_fts, aug_obj_locs, aug_obj_gmm_colors, aug_obj_labels, \
            aug_obj_ids, aug_tgt_obj_idx, aug_obj_ious, aug_obj_loc_gt = self._get_obj_inputs(
                obj_pcds, obj_gmm_colors, obj_labels, obj_ids, tgt_obj_idx, obj_ious, obj_pcd_gt, 
                theta=theta
            )

        aug_obj_fts = torch.from_numpy(aug_obj_fts)
        aug_obj_locs = torch.from_numpy(aug_obj_locs)
        aug_obj_gmm_colors = torch.from_numpy(aug_obj_gmm_colors)
        # aug_obj_classes = torch.LongTensor([self.cat2int[x] for x in aug_obj_labels])
        aug_obj_classes = torch.LongTensor([int(x) for x in aug_obj_labels])
        aug_obj_ious = torch.from_numpy(aug_obj_ious)
        aug_obj_loc_gt = torch.from_numpy(aug_obj_loc_gt)
        
        if self.cat2vec is None:
            aug_obj_gt_fts = aug_obj_classes
        else:
            aug_obj_gt_fts = torch.FloatTensor([self.cat2vec[self.int2cat[int(x)]] for x in aug_obj_labels])

        if 'negative' in item.keys():
            negative = item['negative']
        else:
            negative = 0

        outs = {
            'item_ids': item['item_id'],
            'scan_ids': scan_id,
            'txt_ids': txt_tokens,
            'txt_ids_mask': txt_tokens_mask,
            'txt_lens': txt_lens,
            'txt_lens_mask': txt_lens_mask,
            'obj_gt_fts': aug_obj_gt_fts,
            'obj_fts': aug_obj_fts,
            'obj_locs': aug_obj_locs,
            'obj_colors': aug_obj_gmm_colors,
            'obj_lens': len(aug_obj_fts),
            'obj_classes': aug_obj_classes, 
            'tgt_obj_idxs': aug_tgt_obj_idx,
            'tgt_obj_classes': self.cat2int[tgt_obj_type],
            'obj_ids': aug_obj_ids,
            'change': change_obj,
            'negative': negative, 
            'obj_ious': aug_obj_ious,  
            'tgt_obj_locs_gt': aug_obj_loc_gt, 
        }
        if new_obj_fts:
            outs['new_tgt_feats'] = new_obj_fts
        return outs

def gtlabelpcd_collate_fn(data):
    outs = {}
    for key in data[0].keys():
        outs[key] = [x[key] for x in data]
        
    outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
    outs['txt_ids_mask'] = pad_sequence(outs['txt_ids_mask'], batch_first=True)
    outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
    outs['txt_lens_mask'] = torch.LongTensor(outs['txt_lens_mask'])
    outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])
    outs['txt_masks_mask'] = gen_seq_masks(outs['txt_lens_mask'])

    outs['obj_gt_fts'] = pad_tensors(outs['obj_gt_fts'], lens=outs['obj_lens'])
    outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens'], pad_ori_data=True)
    outs['obj_locs'] = pad_tensors(outs['obj_locs'], lens=outs['obj_lens'], pad=0)
    outs['obj_colors'] = pad_tensors(outs['obj_colors'], lens=outs['obj_lens'], pad=0)
    outs['obj_lens'] = torch.LongTensor(outs['obj_lens'])
    outs['obj_masks'] = gen_seq_masks(outs['obj_lens'])

    outs['obj_ious'] = pad_tensors(outs['obj_ious'], lens=outs['obj_lens'], pad=0)

    outs['obj_classes'] = pad_sequence(
        outs['obj_classes'], batch_first=True, padding_value=-100
    )
    outs['tgt_obj_idxs'] = torch.LongTensor(outs['tgt_obj_idxs'])
    outs['tgt_obj_classes'] = torch.LongTensor(outs['tgt_obj_classes'])

    outs['change'] = torch.LongTensor(outs['change'])
    if 'new_tgt_feats' in outs.keys():
        outs['new_tgt_feats'] = torch.Tensor(outs['new_tgt_feats'])

    outs['negative'] = torch.LongTensor(outs['negative'])
    outs['tgt_obj_locs_gt'] = torch.stack(outs['tgt_obj_locs_gt'])
    return outs


def change_xyz(ori_obj, new_obj):
    # ori_obj, new_obj: N*6
    new_obj_safe = deepcopy(new_obj)
    ori_scale, new_scale, ori_center, new_center = [], [], [], []
    for i in range(3):
        ori_scale.append(np.max(ori_obj[:,i])-np.min(ori_obj[:,i]))
        new_scale.append(np.max(new_obj_safe[:,i])-np.min(new_obj_safe[:,i]))
        ori_center.append((np.max(ori_obj[:,i])+np.min(ori_obj[:,i]))/2)
        new_center.append((np.max(new_obj_safe[:,i])+np.min(new_obj_safe[:,i]))/2)
    ori_center = np.array(ori_center)
    new_center = np.array(new_center)
    scale = max(ori_scale) / max(new_scale)
    new_obj_xyz = (new_obj_safe[:,:3] - new_center) * scale + ori_center
    new_obj_safe[:,:3] = new_obj_xyz

    return new_obj_safe


def calc_3diou(pred_bbox, gt_bbox):

    cx_1, cy_1, cz_1, wx_1, wy_1, wz_1 = pred_bbox
    cx_2, cy_2, cz_2, wx_2, wy_2, wz_2 = gt_bbox
    x_max_1, x_min_1 = cx_1 + wx_1 / 2, cx_1 - wx_1 / 2
    y_max_1, y_min_1 = cy_1 + wy_1 / 2, cy_1 - wy_1 / 2
    z_max_1, z_min_1 = cz_1 + wz_1 / 2, cz_1 - wz_1 / 2
    x_max_2, x_min_2 = cx_2 + wx_2 / 2, cx_2 - wx_2 / 2
    y_max_2, y_min_2 = cy_2 + wy_2 / 2, cy_2 - wy_2 / 2
    z_max_2, z_min_2 = cz_2 + wz_2 / 2, cz_2 - wz_2 / 2
    xA = max(x_min_1, x_min_2)
    yA = max(y_min_1, y_min_2)
    zA = max(z_min_1, z_min_2)
    xB = min(x_max_1, x_max_2)
    yB = min(y_max_1, y_max_2)
    zB = min(z_max_1, z_max_2)
    inter_vol = max((xB - xA), 0) * max((yB - yA), 0) * max((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou