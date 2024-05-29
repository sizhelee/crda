import os
import torch
import numpy as np
from tqdm import trange

pthPath = '/network_space/storage43/lisizhe/dataset/referit3d/scan_data/pcd_with_global_alignment/'
POINTGROUP_PATH = '/network_space/storage43/lisizhe/dataset/scannet/pointgroup_inst/'

outputPath = '/network_space/storage43/lisizhe/dataset/referit3d/scan_data/pcd_with_global_alignment_pred/'

pcds = os.listdir(pthPath)

for i in trange(len(pcds)):

    pcd_file = pcds[i]

    scan_id = pcd_file[:-4]
    if scan_id[-3:] != '_00':
        continue
    if int(scan_id[5:9]) > 706:
        continue

    pcd_path = pthPath + pcd_file
    pcd_data = torch.load(pcd_path)

    coords, colors = pcd_data[0], pcd_data[1]
    instance_labels = pcd_data[-1]

    instance_labels_pred = np.load(POINTGROUP_PATH + 'segv/%s.npy'%scan_id)
    iou_pred = np.load(POINTGROUP_PATH + 'iou/%s.npy'%scan_id)

    torch.save(
        (coords, colors, iou_pred, instance_labels_pred), 
        os.path.join(outputPath, '%s.pth'%(scan_id))
    )