import cv2
from pypcd4 import PointCloud
import numpy as np

from auto_calib_I2I import AutoCalibration



images = []
images_depth = []
pointclouds = []

for i in range(5):
# for i in range(8):
# for i in range(8,10):
    images.append(cv2.imread(f'../data/imgs/{i}.bmp'))
    pointclouds.append(PointCloud.from_path(f"../data/pcds/{i}.pcd").numpy())

init_params = {'x': 0, 'y': 0, 'z': 0, 'roll_deg': 95,'pitch_deg': 0, 'yaw_deg': 95}
config = {
    'camera_intrinsic': np.array([
        [950.7548854113494, 0.0, 790.0352715473131],
        [0.0, 946.9223415597996, 258.3805580551492],
        [0.0, 0.0, 1.0]
    ]) 
}

calib = AutoCalibration(images, pointclouds, init_params, config)

ext_vec_opt = calib.optimize()
image_opt = calib.project_pointcloud_to_image(ext_vec_opt, pointclouds[0], images[0])

ext_vec_init = [init_params['x'], init_params['y'], init_params['z'], init_params['roll_deg'] * np.pi/180, init_params['pitch_deg'] * np.pi/180, init_params['yaw_deg'] * np.pi/180]
image_init = calib.project_pointcloud_to_image(ext_vec_init, pointclouds[0], images[0])
result = cv2.vconcat([image_init, image_opt])
cv2.imshow('', result)
cv2.waitKey(0)

breakpoint()