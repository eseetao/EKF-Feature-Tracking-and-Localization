__author__ = "Erik Seetao"

import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os

from utils import *

class EKF_update:
    def __init__(self, t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu):

        self.b = b
        self.cam_T_imu = cam_T_imu
        self.K = K

        self.inv_oTi = np.linalg.inv(cam_T_imu)
        self.inv_K = np.linalg.inv(K)

    def update(self, z_t, flag, mu_prior, sigma_prior, u_t, sigma_t):
        '''
        update step of EKF
        Input: 
            z_t: (np.array) features per timestamp
            flag: (number of features x 1 np.array) boolean flag vector of features
            mu_prior: (np.array) pose t+1|t predicted from EKF prediction step
            sigma_prior: (np.array) covariance t+1|t predicted from EKF prediction step
            u_t: world coordinates of features in [x y z]
            sigma_t: (np.array) covariance from previous EKF update step

        Output:
            u_t: (3x1 np.array) updated world coordinates of the features in [x y z]
            sigma_t: (3x3 np.array) updated covariance
            flag: (number of features x 1 np.array) updated boolean flag vector of features
        '''

        #predict observation step
        D = np.array( [[1, 0, 0,], [0, 1, 0], [0, 0, 1], [0, 0, 0]]) #dilation matrix 
        feature_len = len(z_t[1])

        #construct M
        fx = self.K[0][0]
        fy = self.K[1][1]
        cx = self.K[0][2]
        cy = self.K[1][2]
        M = np.array([ [fx, 0, cx, 0], [0, fy, cy, 0], [fx, 0, cx, -fx * self.b], [0, fy, cy, 0] ])
        M_inv = np.linalg.pinv(M)
        
        for j in range(feature_len): #j is our landmark index

            if z_t[0,j] == -1:
                #Invalid feature
                continue

            elif flag[j] == False and z_t[0,j] != -1:
                #First time seeing a feature

                #calculate new u_t for particular feature
                flag[j] = True  
                ul_vl_stack = np.array([ z_t[0,j], z_t[1,j], 1 ]).reshape((3,1)) #[ul vl 1] was used in previous calculation of the first feature


                #update mean, but do not update variance

                #using the M^-1 * feature approach
                first_feat = M_inv @ z_t[:,j]
                disparity = first_feat / first_feat[3]

                updated_mu = np.linalg.inv(mu_prior) @ self.inv_oTi @ disparity
                u_t[:,j] = updated_mu.reshape(4) 
                continue
            

            else:
                #On already tracked feature
                optical_T_imu = self.cam_T_imu
                optical = optical_T_imu @ mu_prior @ u_t[:,j] #oTi Tt MUt,j

                #create pi
                q1 = optical[0]
                q2 = optical[1]
                q3 = optical[2]
                q4 = optical[3]
                optical_znorm = optical/q3

                z_hat = M @ optical_znorm #in [ul vl ur vr]
                
                #construct jacobian
                #use same optical oTiTtMUtj
                jacob = np.array([ [1, 0, -q1/q3, 0], [0, 1, -q2/q3, 0], [0, 0, 0, 0], [0, 0, -q4/q3, 1] ])/q3
                H = M @ jacob @ optical_T_imu @ mu_prior @ D

                #perform EKF update
                covariance = sigma_t

                V = np.eye(4) * 0.001 #try noise 0.004

                kalman_gain = covariance @ H.T @ np.linalg.pinv( H @ covariance @ H.T + V ) /100 #squash the inverse noise
                u_t[:,j] = u_t[:,j] + D @ kalman_gain @ ( z_t[:,j] - z_hat ) 

                sigma_t = (np.eye(3) - kalman_gain @ H ) @ covariance


        return u_t, sigma_t, flag
