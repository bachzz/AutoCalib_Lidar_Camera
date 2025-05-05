import cv2
from pypcd4 import PointCloud
import numpy as np

from auto_calib_D2D_vectorized import AutoCalibration

import time


images = []
images_depth = []
pointclouds = []


# for i in range(8):
for i in range(8, 10):
    images.append(cv2.imread(f'../data/imgs/{i}.bmp'))
    images_depth.append(np.load(f'../data/depths/{i}.npy'))
    pointclouds.append(PointCloud.from_path(f"../data/pcds/{i}.pcd").numpy())


init_params = {'x': 0, 'y': 0, 'z': 0, 'roll_deg': 95,'pitch_deg': 0, 'yaw_deg': 95}

gt_params = {'x': 0.001, 'y': 0.018, 'z': 0.063, 'roll_deg': 89.977,'pitch_deg': -0.309, 'yaw_deg': 91.095} 


params_grid = {'x':[], 'y':[], 'z':[], 'roll_deg':[], 'pitch_deg':[], 'yaw_deg':[]}
N = 15

params_grid['x'] = np.linspace(-0.2, 0.2, N) + gt_params['x']
params_grid['y'] = np.linspace(-0.2, 0.2, N) + gt_params['y']
params_grid['z'] = np.linspace(-0.2, 0.2, N) + gt_params['z']
params_grid['roll_deg'] = (np.linspace(-10, 10, N) + gt_params['roll_deg']) * np.pi/180 
params_grid['pitch_deg'] = (np.linspace(-10, 10, N) + gt_params['pitch_deg']) * np.pi/180 
params_grid['yaw_deg'] = (np.linspace(-10, 10, N) + gt_params['yaw_deg']) * np.pi/180 

config = {
    'camera_intrinsic': np.array([
        [950.7548854113494, 0.0, 790.0352715473131],
        [0.0, 946.9223415597996, 258.3805580551492],
        [0.0, 0.0, 1.0]
    ]) 
}


calib = AutoCalibration(images, images_depth, pointclouds, gt_params, config, gt_params=gt_params, params_grid=params_grid)
# calib.plot_MI(flag='010001')
# calib.plot_MI(flag='100001')
# calib.plot_MI(flag='110000')
# calib.plot_MI(flag='010100')
# calib.plot_MI(flag='000101')
# calib.plot_MI(flag='000110')
# calib.plot_MI(flag='000011')


calib = AutoCalibration(images, images_depth, pointclouds, init_params, config, gt_params=gt_params, params_grid=params_grid)

# im_depth = np.tile((images_depth[0] * 255 / images_depth[0].max())[:,:, np.newaxis], (1,1,3))
# ext_vec_init = [init_params['x'], init_params['y'], init_params['z'], init_params['roll_deg'] * np.pi/180, init_params['pitch_deg'] * np.pi/180, gt_params['yaw_deg'] * np.pi/180]
# image_init = calib.project_pointcloud_to_image(ext_vec_init, pointclouds[0], im_depth.astype(np.uint8))
# cv2.imshow('', image_init)
# cv2.waitKey(0)
# breakpoint()


t = time.time()
ext_vec_opt = calib.optimize()
print(f"[optimization] time taken = {time.time() - t}")
image_opt = calib.project_pointcloud_to_image(ext_vec_opt, pointclouds[0], images[0])

ext_vec_init = [init_params['x'], gt_params['y'], init_params['z'], init_params['roll_deg'] * np.pi/180, init_params['pitch_deg'] * np.pi/180, init_params['yaw_deg'] * np.pi/180]
image_init = calib.project_pointcloud_to_image(ext_vec_init, pointclouds[0], images[0])

result = np.zeros([images[0].shape[0]*2, images[0].shape[1]*2, 3], dtype=np.uint8)
result[:images[0].shape[0], images[0].shape[1]:, :] = image_init
result[images[0].shape[0]:, images[0].shape[1]:, :] = image_opt
result[images[0].shape[0]:, :images[0].shape[1], :] = images[0]
cv2.imshow('', result)
cv2.waitKey(0)

breakpoint()