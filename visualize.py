import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import os

info_path = '/home/ljb/data/kitti/'
file_name = 'kitti_infos_val_1.pkl'
info_file_path = info_path + file_name
output_dir = './visualize'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
kitti_info = []

with open(info_file_path, 'rb') as f:
    infos = pickle.load(f)
    kitti_info.extend(infos)

gt_annos = [copy.deepcopy(info['annos']) for info in kitti_info]
gt_annos_pts_nums = []
density = []
distribution = []

zero_points_scenes_path = os.path.join(output_dir, 'zero_points_scenes.txt')
with open(zero_points_scenes_path, 'w') as f:
    for i, gt_anno in enumerate(gt_annos):
        num_obj = len([name for name in gt_anno['name'] if name != 'DontCare'])
        gt_annos_pts_nums.extend(gt_anno['num_points_in_gt'][:num_obj])
        # 在可视化的过程中，我发现居然有500多个框中里面一个点都没有，于是我想记录一下哪些场景内出现了这种情况
        gt_pts_nums = gt_anno['num_points_in_gt'][:num_obj]
        zero_pts_index = [index for index in range(num_obj) if gt_pts_nums[index] == 0]
        if zero_pts_index != [] :
            f.write(f"{kitti_info[i]['point_cloud']['lidar_idx']}\n")
        density.extend(gt_anno['density_points_in_gt'][:num_obj])
        distribution.extend(gt_anno['distribution_points_in_gt'][:num_obj])



# 绘制并保存第一张图片
plt.figure()
counts, bin_edges = np.histogram(gt_annos_pts_nums, bins=200, range=(0, 2000))
cumulative_counts = np.cumsum(counts)
plt.plot(bin_edges[:-1], cumulative_counts, drawstyle='steps-post')
plt.title('Histogram of Ground Truth Points')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_dir, 'histogram_gt_points.png'))
plt.close()

# 绘制并保存第二张图片
plt.figure()
counts, bin_edges = np.histogram(density, bins=300, range=(0, 150))
cumulative_counts = np.cumsum(counts)
plt.plot(bin_edges[:-1], cumulative_counts, drawstyle='steps-post')
plt.title('Histogram of Density')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_dir, 'histogram_density.png'))
plt.close()

# 绘制并保存第三张图片
plt.figure()
counts, bin_edges = np.histogram(distribution, bins=125, range=(0, 1))
cumulative_counts = np.cumsum(counts)
plt.plot(bin_edges[:-1], cumulative_counts, drawstyle='steps-post')
plt.title('Histogram of Distribution')
plt.xlabel('Distribution')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_dir, 'histogram_distribution.png'))
plt.close()


