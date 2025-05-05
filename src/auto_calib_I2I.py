import numpy as np
# from sklearn.metrics import mutual_info_score

from scipy.spatial.transform import Rotation as R

#from util import project_pointcloud_on_image, project_pointcloud_on_image2
import cv2
import time


D2R = np.pi / 180 # degree to radian const
DIST_THRES = 10000


class Util():
    def __init__(self):
        return
        
    
    def project_point_to_image(self, P_L, T, cam_K):
        P_C = T @ np.array([[P_L[0], P_L[1], P_L[2], 1]]).T
        p_C = cam_K @ P_C[:3]
        if np.unique(p_C)[0] == 0.0: return np.nan
        
        return [int(p_C[0][0]/p_C[2][0]), int(p_C[1][0]/p_C[2][0]) ]
    
    
    def validate_image_point(self, p_C, image):

        return not np.isnan(p_C).any() and p_C[0] > 0 and p_C[0] < image.shape[1] and p_C[1] > 0 and p_C[1] < image.shape[0]
    
    
    def project_pointcloud_to_image(self, pointcloud, image, ext_vec, cam_K):
        ###
        ### ext_vec: x, y, z, roll_rad, pitch_rad, yaw_rad (zyx euler-angle, radian)
        ###
        
        image_ = image.copy()
        projected_points = []
        
        for point in pointcloud:
            px, py, pz, pi = point
            point_dist = px**2 + py**2 + pz**2
            
            if (point_dist / DIST_THRES) < 1: # avoid points in inifite
                ### project point into image
                t =  ext_vec[:3]
                rot = R.from_euler('zyx', ext_vec[3:])
                T = np.eye(4); T[:3, :3] = rot.as_matrix(); T[:3,3] = t
                
                P_L = [px, py, pz]
                p_C = self.project_point_to_image(P_L, T, cam_K)
                
                ### validate point
                if not self.validate_image_point(p_C, image):
                    continue
                
                projected_points.append({'coord':p_C, 'intensity': pi})
                
        ### draw projected points
        image_ = self.draw_pointcloud_on_image(projected_points, image_)
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
    
    def draw_pointcloud_on_image(self, projected_points, image):
        for point in projected_points:
            image = cv2.circle(image, center=point['coord'], radius=1, color=self.colorCodingReflectivity(point['intensity']), thickness=-1)
        
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
            
        
    

class AutoCalibration():
    def __init__(self, images, pointclouds, init_params, config, max_iters = 300):
        
        ### general
        self.init_params = init_params
        self.images_bgr = images 
        self.images = []
        for image in images:
            self.images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        self.pointclouds = pointclouds
        
        self.N = len(self.images)
        
        self.cam_K = config['camera_intrinsic']
        
        ### extrinsic matrix
        self.ext_vec_ = None
        
        
        ### gradient descent params     
        # step size params
        self.gamma_trans_ = 0.01 # translation
        self.gamma_trans_u_ = 0.1 # upper bound
        self.gamma_trans_l_ = 0.001 # lower bound
        
        self.gamma_rot_ = 0.001 # rotation
        self.gamma_rot_u_ = 0.05 # upper bound
        self.gamma_rot_l_ = 0.0005 # lower bound
        
        self.eps_ = 1e-9
        
        # finite increments
        self.delta_ = np.array([0.01, 0.01, 0.01, 0.1 * D2R, 0.1 * D2R, 0.1 * D2R]) # x, y, z, r, p, y
        
        # max inters
        self.max_iters_ = max_iters
        
        
        ### misc
        self.MAX_BINS = 256
        self.num_bins = 256
        self.bin_fraction = self.MAX_BINS / self.num_bins
        
        ### support
        self.hist_handler = HistogramHandler(self.num_bins)
        self.utils = Util()
        
    
    def project_pointcloud_to_image(self, ext_vec, pointcloud, image):
        return self.utils.project_pointcloud_to_image(pointcloud, image, ext_vec, self.cam_K) 

    
    def compute_hist(self, ext_vec):
        ###
        ### ext_vec: x, y, z, roll_rad, pitch_rad, yaw_rad (zyx euler-angle, radian)
        ###
        
        self.hist_handler.reset()
        
        for i in range(self.N):
            for point in self.pointclouds[i]:
                px, py, pz, pi = point
                point_dist = px**2 + py**2 + pz**2
                
                if (point_dist / DIST_THRES) < 1: # avoid points in inifite
                    ### project point into image
                    t =  ext_vec[:3]
                    rot = (R.from_rotvec(ext_vec[3] * np.array([1,0,0])) * R.from_rotvec(ext_vec[4] * np.array([0,1,0])) * R.from_rotvec(ext_vec[5] * np.array([0,0,1])) )
                    T = np.eye(4); T[:3, :3] = rot.as_matrix(); T[:3,3] = t
                    
                    P_L = [px, py, pz]
                    p_C = self.utils.project_point_to_image(P_L, T, self.cam_K)
                    
                    ### validate point
                    if not self.utils.validate_image_point(p_C, self.images[i]):
                        continue
                    
                    intensity_bin = int(pi / self.bin_fraction)
                    gray_bin = int(self.images[i][p_C[1], p_C[0]] / self.bin_fraction)
                    
                    ### update hist
                    self.hist_handler.intensity_hist[intensity_bin] += 1
                    self.hist_handler.gray_hist[gray_bin] += 1
                    self.hist_handler.joint_hist[gray_bin, intensity_bin] += 1
                    
                    self.hist_handler.intensity_sum += intensity_bin
                    self.hist_handler.gray_sum += gray_bin
                    
                    self.hist_handler.total_points += 1
            
        
    def estimate_MLE(self):
        prob_intensity = self.hist_handler.intensity_hist / self.hist_handler.total_points
        prob_gray = self.hist_handler.gray_hist / self.hist_handler.total_points
        prob_joint = self.hist_handler.joint_hist / self.hist_handler.total_points
        
        ### smoothing with KDE
        sigma_intensity, sigma_gray = self.hist_handler.compute_stds()
        
        ### bandwidth for KDE based on Silverman's rule of thumb
        sigma_intensity_bandwidth = 1.06 * np.sqrt(sigma_intensity) / (self.hist_handler.total_points ** 2)
        sigma_gray_bandwidth = 1.06 * np.sqrt(sigma_gray) / (self.hist_handler.total_points ** 2)
        
        prob_intensity = cv2.GaussianBlur(prob_intensity, (0, 0), sigmaX=sigma_intensity_bandwidth)
        prob_gray = cv2.GaussianBlur(prob_gray, (0, 0), sigmaX=sigma_gray_bandwidth)
        prob_joint = cv2.GaussianBlur(prob_joint, (0, 0), sigmaX=sigma_gray_bandwidth, sigmaY=sigma_intensity_bandwidth)
        
        return prob_intensity, prob_gray, prob_joint
    
    
    def compute_MI(self, ext_vec, normalize = False):
        ### compute hist
        self.compute_hist(ext_vec)
        
        ### compute probs
        prob_intensity, prob_gray, prob_joint = self.estimate_MLE()
        
        prob_intensity_masked = prob_intensity[prob_intensity != 0]
        prob_gray_masked = prob_gray[prob_gray != 0]
        prob_joint_masked = prob_joint[prob_joint != 0]
        # breakpoint()
        
        ### compute entropies
        H_intensity = -(prob_intensity_masked * np.log2(prob_intensity_masked) ).sum()
        H_gray = -(prob_gray_masked * np.log2(prob_gray_masked)).sum()
        H_joint = -(prob_joint_masked * np.log2(prob_joint_masked)).sum()
        
        ### compute MI
        mi_score = H_intensity + H_gray - H_joint
        mi_score_norm = 2 * mi_score / (H_intensity + H_gray)
        
        # print(f'mi_score = {mi_score}')
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
            
            print(f"f_prev = {f_prev}")
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
                
                if self.delta_[0] < 0.001:
                    self.ext_vec_ = ext_vec
                    break
                else:
                    continue
            
            ### update prev gradients
            grad_x_prev, grad_y_prev, grad_z_prev, grad_roll_prev, grad_pitch_prev, grad_yaw_prev = grad_x, grad_y, grad_z, grad_roll, grad_pitch, grad_yaw
            
            print(f"[iter={idx}] f = {f} - ext_vec = {ext_vec * [1, 1, 1, 1/D2R, 1/D2R, 1/D2R] } - time = {time.time() - t}")
          
            
        print(f"[optimized] ext_vec = {ext_vec * [1, 1, 1, 1/D2R, 1/D2R, 1/D2R] }")
        return self.ext_vec_
        