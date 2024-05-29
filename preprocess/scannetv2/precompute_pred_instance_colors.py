import os
import json
import glob
import torch
import numpy as np

from sklearn.mixture import GaussianMixture


scan_dir = '/network_space/storage43/lisizhe/dataset/referit3d/scan_data/'
output_dir = os.path.join(scan_dir, 'instance_id_to_gmm_color_pred')
os.makedirs(output_dir, exist_ok=True)

for scan_file in glob.glob(os.path.join(scan_dir, 'pcd_with_global_alignment_pred', '*')):
    scan_id = os.path.basename(scan_file).split('.')[0]
    print(scan_file)
    
    data = torch.load(scan_file) # xyz, rgb, semantic_labels, instance_labels
    colors = data[1]
    instance_labels = data[3]

    if instance_labels is None:
        continue

    # normalize
    colors = colors / 127.5 - 1 
    
    clustered_colors = []
    for i in range(instance_labels.max() + 1):
        mask = instance_labels == i     # time consuming
        obj_colors = colors[mask]

        if obj_colors.shape[0] < 3:
            obj_colors = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        
        gm = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(obj_colors)
        clustered_colors.append({
            'weights': gm.weights_.tolist(),
            'means': gm.means_.tolist(),
        })
        
    json.dump(
        clustered_colors,
        open(os.path.join(output_dir, '%s.json'%scan_id), 'w')
    )
