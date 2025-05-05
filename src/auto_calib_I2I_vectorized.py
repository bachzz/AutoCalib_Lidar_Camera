import numpy as np
# from sklearn.metrics import mutual_info_score

from scipy.spatial.transform import Rotation as R

#from util import project_pointcloud_on_image, project_pointcloud_on_image2
import matplotlib.pyplot as plt
import cv2
import time

# # https://forum.qt.io/post/654289
# import os
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
# os.environ.pop("QT_QPA_FONTDIR")


np.set_printoptions(precision=3, suppress=True)


D2R = np.pi / 180 # degree to radian const
DIST_THRES = 10000


class Util():
    def __init__(self):
        return
        
    
    def project_point_to_image(self, P_L, T, cam_K):
        # breakpoint()
        P_C = T @ np.array([[P_L[0], P_L[1], P_L[2], 1]]).T
        p_C = cam_K @ P_C[:3]
        # breakpoint()
        if np.unique(p_C)[0] == 0.0: return [np.nan, np.nan]
        # if p_C[2][0] < 0:
        #     print(p_C[2][0])
        return [int(p_C[0][0]/p_C[2][0]), int(p_C[1][0]/p_C[2][0]) ]
    
    
    def validate_image_point(self, p_C, image):
        # if p_C == np.nan: return False
        # breakpoint()
        return not np.isnan(p_C).any() and p_C[0] > 0 and p_C[0] < image.shape[1] and p_C[1] > 0 and p_C[1] < image.shape[0]
    
    
    def project_pointcloud_to_image(self, pointcloud, image, ext_vec, cam_K):
        ###
        ### ext_vec: x, y, z, roll_rad, pitch_rad, yaw_rad (xyz euler-angle, radian)
        ### rotate by x -> y -> z (intrinsic rotation)
        
        image_ = image.copy()
        
        t = np.array([ext_vec[:3]]).T
        # rot_mat = R.from_euler('zyx', ext_vec[3:]).as_matrix()
        #rot_mat = R.from_euler('xyz', ext_vec[3:]).as_matrix() # XYZ (intrinsic), xyz (extrinsic)
        rot_mat = (R.from_rotvec(ext_vec[3] * np.array([1,0,0])) * R.from_rotvec(ext_vec[4] * np.array([0,1,0])) * R.from_rotvec(ext_vec[5] * np.array([0,0,1])) ).as_matrix()
        pointcloud_C = rot_mat @ pointcloud.T[:3,:] + t
        
        p_C = cam_K @ pointcloud_C
        p_C = p_C / p_C[2, :] # normalize with z
        p_C = p_C[:2,:]
        
        delta_C = p_C - np.array([[image.shape[1], image.shape[0]]]).T # check if pixel within image
        
        mask_C = (delta_C < 0) & (p_C > 0)
        mask_C = mask_C[0,:] & mask_C[1,:]
        
        p_C_intensity = np.append(p_C, np.array([pointcloud.T[3,:]]), axis=0)
        p_C_intensity_masked = np.where(mask_C, p_C_intensity, np.nan)
        p_C_intensity_masked = p_C_intensity_masked[:, ~np.isnan(p_C_intensity_masked).all(axis=0)]
        
        image_ = self.draw_pointcloud_on_image(p_C_intensity_masked.T, image_)
        
        return image_
    
    
    def colorCodingReflectivity(self, intensity):
        r, g, b = 0, 0, 0
        if intensity < 30:
            r = 0
            g = int(intensity * 255 / 30) & 0xFF
            b = 255
        elif intensity < 90:
            r = 0
            g = 0xFF
            b = int((90 - intensity) * 255 / 60 ) & 0xFF
        elif intensity < 150:
            r = int((intensity - 90) * 255 / 60 ) & 0xFF
            g = 0xFF
            b = 0
        else:
            r = 0xFF
            g = int((255-intensity) * 255 / (256-150) ) & 0xFF
            b = 0
        
        return (b, g, r)
    
    # def draw_pointcloud_on_image(self, projected_points, image):
    #     for point in projected_points:
    #         image = cv2.circle(image, center=point['coord'], radius=1, color=self.colorCodingReflectivity(point['intensity']), thickness=-1)
        
    #     return image
    
    def draw_pointcloud_on_image(self, projected_points, image):
        for point in projected_points:
            # print(point)
            image = cv2.circle(image, center=(int(point[0]), int(point[1])), radius=1, color=self.colorCodingReflectivity(point[2]), thickness=-1)
        
        return image


class HistogramHandler():
    def __init__(self, num_bins):
        self.num_bins = num_bins
        
        self.intensity_hist = None
        self.gray_hist = None
        self.joint_hist = None
    
        self.intensity_sum = None
        self.gray_sum = None
        
        self.total_points = None
        
        self.reset()
    
    
    def reset(self):
        self.intensity_hist = np.zeros(self.num_bins)
        self.gray_hist = np.zeros(self.num_bins)
        self.joint_hist = np.zeros([self.num_bins, self.num_bins])
    
        self.intensity_sum = 0
        self.gray_sum = 0
        
        self.total_points = 0
        
    def compute_stds(self):
        intensity_mean = self.intensity_sum / self.total_points
        gray_mean = self.gray_sum / self.total_points
        
        intensity_sigma = 0
        gray_sigma = 0
        for i in range(self.num_bins):
            intensity_sigma += self.intensity_hist[i] * (i - intensity_mean) ** 2
            gray_sigma += self.gray_hist[i] * (i - gray_mean) ** 2
        intensity_sigma = np.sqrt(intensity_sigma / self.total_points)
        gray_sigma = np.sqrt(gray_sigma / self.total_points)
        
        return intensity_sigma, gray_sigma
            
        

# plt.ion()
# fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
# plt.show()

class AutoCalibration():
    def __init__(self, images, pointclouds, init_params, config, max_iters = 300, gt_params=None, params_grid=None):
        
        ### general
        self.init_params = init_params
        self.images_bgr = images #image
        self.images = [] #images_depth #[] #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in images:
            self.images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        self.gt_params = gt_params
        self.params_grid = params_grid
        
        # for image in images_depth:
        #     image = (image * 255 / image.max()).astype(np.uint8) 
        #     image = cv2.medianBlur(image, 31)
        #     self.images.append(image)
        
        # for image in images_depth: #images:
        #     self.images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            
        self.image_H, self.image_W = self.images[0].shape
            
        self.pointclouds = pointclouds
        
        self.N = len(self.images)
        
        self.cam_K = config['camera_intrinsic']
        
        ### extrinsic matrix
        self.ext_vec_ = None
        
        
        ### gradient descent params     
        # step size params
        self.gamma_trans_ = 0.01 # translation
        self.gamma_trans_u_ = 0.1 # upper bound
        self.gamma_trans_l_ = 0.001 #0.01 #0.001 # lower bound
        
        self.gamma_rot_ = 0.001 # rotation
        self.gamma_rot_u_ = 0.05 #3.14 * np.pi/180 #0.05 # upper bound
        self.gamma_rot_l_ = 0.0005 #0.1 * np.pi/180 #0.0005 # lower bound
        
        self.eps_ = 1e-9
        
        # finite increments
        self.delta_ = np.array([0.01, 0.01, 0.01, 0.1 * D2R, 0.1 * D2R, 0.1 * D2R]) # x, y, z, r, p, y
        self.delta_thres = 0.0001
        
        # max inters
        self.max_iters_ = max_iters
        
        
        ### misc
        self.MAX_BINS = 256 #150 #256
        self.num_bins = 256 #150 #256
        self.bin_fraction = self.MAX_BINS / self.num_bins
        
        ### support
        self.hist_handler = HistogramHandler(self.num_bins)
        self.utils = Util()
        
        
    def plot_MI(self, flag):  
        flag_idxes = np.where(np.array(list(flag)) == '1')[0]
              
        X = list(self.params_grid.values())[flag_idxes[0]]
        Y = list(self.params_grid.values())[flag_idxes[1]]
        X_, Y_ = np.meshgrid(X, Y)
        # breakpoint()
        
        def compute_MI_lambda(x, y, flag): #flag):
            flag_idxes = np.where(np.array(list(flag)) == '1')[0]
            # breakpoint()
            ext_vec = np.array([self.gt_params['x'], self.gt_params['y'], self.gt_params['z'], self.gt_params['roll_deg'] * D2R, self.gt_params['pitch_deg'] * D2R, self.gt_params['yaw_deg'] * D2R ])
            ext_vec[flag_idxes[0]] = x
            ext_vec[flag_idxes[1]] = y
            return self.compute_MI(ext_vec)
        
        mi_scores = np.vectorize(compute_MI_lambda)(X_, Y_, flag)
        
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}); ax.plot_surface(X_, Y_ * (1/D2R), mi_scores, cmap=matplotlib.cm.jet); plt.show()
        plt.colorbar(plt.pcolor(X_ * (1/D2R),Y_ * (1/D2R),mi_scores)); plt.show()
        # plt.colorbar(plt.pcolor(X_,Y_,mi_scores)); plt.show()
        
        breakpoint()
    
    
    def project_pointcloud_to_image(self, ext_vec, pointcloud, image):
        return self.utils.project_pointcloud_to_image(pointcloud, image, ext_vec, self.cam_K) 

    
    def compute_hist(self, ext_vec):
        ###
        ### ext_vec: x, y, z, roll_rad, pitch_rad, yaw_rad (zyx euler-angle, radian)
        ###
        
        self.hist_handler.reset()
        
        for i in range(self.N):
            ### project P_L to P_C
            t = np.array([ext_vec[:3]]).T
            rot_mat = (R.from_rotvec(ext_vec[3] * np.array([1,0,0])) * R.from_rotvec(ext_vec[4] * np.array([0,1,0])) * R.from_rotvec(ext_vec[5] * np.array([0,0,1])) ).as_matrix()
            pointcloud_C = rot_mat @ self.pointclouds[i].T[:3,:] + t
            
            ### project P_C to p_C
            p_C = self.cam_K @ pointcloud_C
            p_C_ = p_C / p_C[2, :] # normalize with z
            p_C_ = p_C_[:2,:]
            
            p_C_[np.isnan(p_C_)] = 999999 # divided by 0 -> outliers
            p_C_ = p_C_.astype(int)
            
            delta_C = p_C_ - np.array([[self.image_W, self.image_H]]).T # check if pixel within image
            
            ### check: 1) wihin image, 2) non-negative pixel coord, 3) in front of camera
            mask_C = (delta_C < 0) & (p_C_ >= 0) & (p_C[2, :] > 0)
            mask_C = mask_C[0,:] & mask_C[1,:]
            
            p_C_intensity = np.append(p_C_, np.array([self.pointclouds[i].T[3,:]]), axis=0)
            p_C_intensity_masked = np.where(mask_C, p_C_intensity, np.nan)
            p_C_intensity_masked = p_C_intensity_masked[:, ~np.isnan(p_C_intensity_masked).any(axis=0)] 
            
            p_C_masked = p_C_intensity_masked[:2,:].astype(int)
            intensity_masked = p_C_intensity_masked[2,:]
            
            
            ### compute hist
            gray_masked_bin = self.images[i][p_C_masked[1,:], p_C_masked[0,:]] / self.bin_fraction
            intensity_masked_bin = intensity_masked / self.bin_fraction
            
            bins = self.num_bins 
            self.hist_handler.gray_hist += np.histogram(gray_masked_bin, bins=bins)[0]
            self.hist_handler.intensity_hist += np.histogram(intensity_masked_bin, bins=bins)[0]
            self.hist_handler.joint_hist += np.histogram2d(gray_masked_bin, intensity_masked_bin, bins=(bins, bins))[0]
            
            self.hist_handler.gray_sum += gray_masked_bin.sum()
            self.hist_handler.intensity_sum += intensity_masked_bin.sum()
            
            self.hist_handler.total_points += len(gray_masked_bin)
            
            
        
    def estimate_MLE(self):
        prob_intensity = self.hist_handler.intensity_hist / self.hist_handler.total_points
        prob_gray = self.hist_handler.gray_hist / self.hist_handler.total_points
        prob_joint = self.hist_handler.joint_hist / self.hist_handler.total_points
        
        ### smoothing with KDE
        sigma_intensity, sigma_gray = self.hist_handler.compute_stds()
        
        ### bandwidth for KDE based on Silverman's rule of thumb
        sigma_intensity_bandwidth = 1.06 * np.sqrt(sigma_intensity) / (self.hist_handler.total_points ** 2)
        sigma_gray_bandwidth = 1.06 * np.sqrt(sigma_gray) / (self.hist_handler.total_points ** 2)
        
        # breakpoint()
        prob_intensity = cv2.GaussianBlur(prob_intensity, (0, 0), sigmaX=sigma_intensity_bandwidth)
        prob_gray = cv2.GaussianBlur(prob_gray, (0, 0), sigmaX=sigma_gray_bandwidth)
        prob_joint = cv2.GaussianBlur(prob_joint, (0, 0), sigmaX=sigma_gray_bandwidth, sigmaY=sigma_intensity_bandwidth)
        
        return prob_intensity, prob_gray, prob_joint
    
    
    def compute_MI(self, ext_vec, normalize = False):
        # print(f"ext_vec = {ext_vec * [1, 1, 1, 1/D2R, 1/D2R, 1/D2R] }")
        ### compute hist
        self.compute_hist(ext_vec)
        
        ### compute probs
        prob_intensity, prob_gray, prob_joint = self.estimate_MLE()
        
        prob_intensity_masked = prob_intensity[prob_intensity != 0]
        prob_gray_masked = prob_gray[prob_gray != 0]
        prob_joint_masked = prob_joint[prob_joint != 0]
        
        ### compute entropies
        H_intensity = -(prob_intensity_masked * np.log2(prob_intensity_masked) ).sum()
        H_gray = -(prob_gray_masked * np.log2(prob_gray_masked)).sum()
        H_joint = -(prob_joint_masked * np.log2(prob_joint_masked)).sum()
        
        ### compute MI
        mi_score = H_intensity + H_gray - H_joint
        mi_score_norm = 2 * mi_score / (H_intensity + H_gray)
        
        # print(f'mi_score = {mi_score} - ext_vec = {ext_vec * [1, 1, 1, 1/D2R, 1/D2R, 1/D2R] }')
        
        # breakpoint()
        
        # X, Y = np.meshgrid(np.arange(0,self.num_bins), np.arange(0,self.num_bins)); plt.subplots(subplot_kw={'projection':'3d'})[1].plot_surface(X, Y, self.hist_handler.joint_hist, cmap=plt.cm.viridis);plt.show()
        
        return mi_score_norm if normalize else mi_score
    
    
    def optimize(self):
        ###
        ### optimize ext_pose_ using Borwein (1988) gradient-descent approach
        ### ref: https://robots.engin.umich.edu/SoftwareData/ExtrinsicCalib
        ###
        
        
        ### initialize vars
        
        # max cost
        f_max = 0
        
        # extrinsic matrix
        ext_vec_prev = np.array([self.init_params['x'], self.init_params['y'], self.init_params['z'], self.init_params['roll_deg'] * D2R, self.init_params['pitch_deg'] * D2R, self.init_params['yaw_deg'] * D2R ])
        ext_vec = ext_vec_prev.copy()
        # print(f"ext_vec = {ext_vec}")
        print(f"ext_vec = {ext_vec * [1, 1, 1, 1/D2R, 1/D2R, 1/D2R] }")
        
        # previous gradients
        grad_x_prev, grad_y_prev, grad_z_prev, grad_roll_prev, grad_pitch_prev, grad_yaw_prev = 0, 0, 0, 0, 0, 0
        
        ### optimization loop
        for idx in range(self.max_iters_):
            t = time.time()
            
            ### compute normalized gradients (in Eq. 15) for each component in ext mat
            
            # prev cost
            f_prev = self.compute_MI(ext_vec)
            if f_prev > f_max:
                f_max = f_prev
            
            # print(f"f_prev = {f_prev}")
            
            # increment & compute new cost for each component
            delta_x = ext_vec + np.array([self.delta_[0], 0, 0, 0, 0, 0]) 
            f = self.compute_MI(delta_x)
            grad_x = (f - f_prev) / self.delta_[0]
            
            delta_y = ext_vec + np.array([0, self.delta_[1], 0, 0, 0, 0])
            f = self.compute_MI(delta_y)
            grad_y = (f - f_prev) / self.delta_[1]
            
            delta_z = ext_vec + np.array([0, 0, self.delta_[2], 0, 0, 0])
            f = self.compute_MI(delta_z)
            grad_z = (f - f_prev) / self.delta_[2]
            
            delta_roll = ext_vec + np.array([0, 0, 0, self.delta_[3], 0, 0])
            f = self.compute_MI(delta_roll)
            grad_roll = (f - f_prev) / self.delta_[3]
            
            delta_pitch = ext_vec + np.array([0, 0, 0, 0, self.delta_[4], 0])
            f = self.compute_MI(delta_pitch)
            grad_pitch = (f - f_prev) / self.delta_[4]
            
            delta_yaw = ext_vec + np.array([0, 0, 0, 0, 0, self.delta_[5]])
            f = self.compute_MI(delta_yaw)
            grad_yaw = (f - f_prev) / self.delta_[5]
            
            # normalizing gradients
            # print(grad_x, grad_y, grad_z,grad_roll, grad_pitch, grad_yaw)
            grad_x = grad_x / np.linalg.norm([grad_x, grad_y, grad_z])
            grad_y = grad_y / np.linalg.norm([grad_x, grad_y, grad_z])
            grad_z = grad_z / np.linalg.norm([grad_x, grad_y, grad_z])
            grad_roll = grad_roll / (np.linalg.norm([grad_roll, grad_pitch, grad_yaw]) + self.eps_)
            grad_pitch = grad_pitch / (np.linalg.norm([grad_roll, grad_pitch, grad_yaw]) + self.eps_)
            grad_yaw = grad_yaw / (np.linalg.norm([grad_roll, grad_pitch, grad_yaw]) + self.eps_)
            
            
            ### compute adative step size (separately for trans, rot)
            delta_ext_vec_trans = np.array([ext_vec[0], ext_vec[1], ext_vec[2]]) - np.array([ext_vec_prev[0], ext_vec_prev[1], ext_vec_prev[2]] )
            if np.sum(delta_ext_vec_trans ** 2) > 0:
                delta_grad_trans = np.array([grad_x, grad_y, grad_z]) - np.array([grad_x_prev, grad_y_prev, grad_z_prev])
                self.gamma_trans_ = np.sum(delta_ext_vec_trans ** 2) / (np.abs(np.array([delta_ext_vec_trans]) @ np.array([delta_grad_trans]).T )[0][0] + self.eps_)
            else:
                self.gamma_trans_ = self.gamma_trans_u_
            
            delta_ext_vec_rot = np.array([ext_vec[3], ext_vec[4], ext_vec[5]]) - np.array([ext_vec_prev[3], ext_vec_prev[4], ext_vec_prev[5] ])
            if np.sum(delta_ext_vec_rot ** 2) > 0:
                delta_grad_rot = np.array([grad_roll, grad_pitch, grad_yaw]) - np.array([grad_roll_prev, grad_pitch_prev, grad_yaw_prev])
                self.gamma_rot_ = np.sum(delta_ext_vec_rot ** 2) / (np.abs(np.array([delta_ext_vec_rot]) @ np.array([delta_grad_rot]).T )[0][0] + self.eps_)
            else:
                self.gamma_rot_ = self.gamma_rot_u_
            
            # bounded
            if self.gamma_trans_ > self.gamma_trans_u_:
                self.gamma_trans_ = self.gamma_trans_u_
            if self.gamma_trans_ < self.gamma_trans_l_:
                self.gamma_trans_ = self.gamma_trans_l_
        
            if self.gamma_rot_ > self.gamma_rot_u_:
                self.gamma_rot_ = self.gamma_rot_u_
            if self.gamma_rot_ < self.gamma_rot_l_:
                self.gamma_rot_ = self.gamma_rot_l_
        
            
            ### store ext_vec into prev
            ext_vec_prev = ext_vec.copy()
            
            ### update ext_vec
            #breakpoint()
            # print(self.gamma_trans_ * grad_x,self.gamma_trans_ * grad_y,self.gamma_trans_ * grad_z,self.gamma_rot_ * grad_roll,self.gamma_rot_ * grad_pitch,self.gamma_rot_ * grad_yaw)
            delta_mat = np.array([
                        self.gamma_trans_ * grad_x,
                        self.gamma_trans_ * grad_y,
                        self.gamma_trans_ * grad_z,
                        self.gamma_rot_ * grad_roll,
                        self.gamma_rot_ * grad_pitch,
                        self.gamma_rot_ * grad_yaw
                        ])
            # print(delta_mat)
            ext_vec = ext_vec + delta_mat
            
            
            ### compute new cost
            f = self.compute_MI(ext_vec)
            
            ### if cost decreases -> rollback & adjust step size (more conservative)
            if f < f_prev:
                delta_mat = np.array([
                        self.gamma_trans_ * grad_x,
                        self.gamma_trans_ * grad_y,
                        self.gamma_trans_ * grad_z,
                        self.gamma_rot_ * grad_roll,
                        self.gamma_rot_ * grad_pitch,
                        self.gamma_rot_ * grad_yaw
                        ])
                ext_vec = ext_vec - delta_mat
                self.gamma_trans_u_ = self.gamma_trans_u_ / 1.2
                self.gamma_trans_l_ = self.gamma_trans_l_ / 1.2
                self.gamma_rot_u_ = self.gamma_rot_u_ / 1.2
                self.gamma_rot_l_ = self.gamma_rot_l_ / 1.2
            
                self.delta_ = self.delta_ / 1.1
                
                if self.delta_[0] < self.delta_thres:
                    self.ext_vec_ = ext_vec
                    break
                else:
                    continue
            
            ### update prev gradients
            grad_x_prev, grad_y_prev, grad_z_prev, grad_roll_prev, grad_pitch_prev, grad_yaw_prev = grad_x, grad_y, grad_z, grad_roll, grad_pitch, grad_yaw
            
            print(f"[iter={idx}] f = {f} - ext_vec = {ext_vec * [1, 1, 1, 1/D2R, 1/D2R, 1/D2R] } - duration = {time.time() - t}")
          
            
        print(f"[optimized] ext_vec = {ext_vec * [1, 1, 1, 1/D2R, 1/D2R, 1/D2R] }")
        
        X, Y = np.meshgrid(np.arange(0,self.num_bins), np.arange(0,self.num_bins)); plt.subplots(subplot_kw={'projection':'3d'})[1].plot_surface(X, Y, self.hist_handler.joint_hist, cmap=plt.cm.viridis);plt.show()
        breakpoint()
        return self.ext_vec_
        