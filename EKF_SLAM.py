__author__ = "Erik Seetao"

import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os

from utils import *
from EKF_predict import EKF_predict
from EKF_update import EKF_update

class EKF_SLAM:
    def __init__(self,EKF_predict,EKF_update):
        self.prediction = EKF_predict
        self.update = EKF_update

    def __call__(self, t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu):        

        #initialize
        mu_current = np.eye(4) #mu_0|0
        sigma_current = np.eye(6) #sigma_0|0

        num_features = len(features[1])
        print(num_features)
        u_t = np.tile(np.array([0,0,0,1]), (num_features,1)).T
        sigma_t = np.array([[1,0,0], [0,1,0], [0,0,1]],dtype=float)


        prev_z_t = np.ones((4,num_features)) * -1 #create first instance for seeing a feature
        #will be 4x104 for 0042 and 4x206 for 0027
        flag = [False] * num_features

        pose = mu_current
        total_u_t = prev_z_t 

        for index in range(linear_velocity.shape[1] - 1):
            print("this is the ",index," iteration of the timestamp")
            tau = t[:,index + 1] - t[:,index] #delta tau
            
            #IMU Localization via EKF Prediction
            mu_predict,sigma_predict = EKF_predict.predict(linear_velocity[:,index], rotational_velocity[:,index], tau, mu_current, sigma_current)

            #Landmark Mapping via EKF Update
            z_t = features[:,:,index] #z_t is features per timestamp

            u_t_homog, sigma_t, flag = EKF_update.update(z_t, flag, mu_predict, sigma_predict, u_t, sigma_t)
            #u_t_homog is homogeneous coord
            prev_z_t = z_t #to keep track of if we have seen a feature before

            #update variables
            mu_current = mu_predict #update the prior
            total_u_t = np.dstack((total_u_t, u_t_homog))

            pose_inv = self.pose_inverse(mu_predict) #returns a 4x4 inverse
            pose = np.dstack((pose, pose_inv)) #appending into N, 4x4xN wise
            

        #visualize does imu_T_world, which is our pose_inverse (T^-1)
        visualize_trajectory_2d(pose, total_u_t, show_ori=True)
        




    def pose_inverse(self, mu):
        '''
        finds the inverse pose (T^-1) of SE(3)
        Input:
            mu: (4x4 np array) is mu from EKF predict step
        Output:
            inverse pose (4x4 np array)
        '''
        R = mu[0:3,0:3]
        p = mu[0:3,3].reshape((3,1))

        p_inv = np.matmul(np.transpose(R),p)
        pre_pose = np.append(np.transpose(R), -p_inv,axis=1) #check here in case it's dot product or something else

        pose_inv = np.append(pre_pose, np.array( [0, 0, 0, 1]).reshape((1,4)), axis=0)

        return pose_inv


if __name__ == "__main__":

    data_path = "/Users/eseetao/Documents/School Docs/ECE276A/Project 3/data/"

    filename = data_path + "0042.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)

    EKF_predict = EKF_predict()
    EKF_update = EKF_update(t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu)

    SLAM = EKF_SLAM(EKF_predict,EKF_update)
    SLAM(t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu)


    
