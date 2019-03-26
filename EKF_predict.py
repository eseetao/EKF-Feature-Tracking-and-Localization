__author__ = "Erik Seetao"

import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

from utils import *
from scipy.linalg import expm 

class EKF_predict:
    def __init__(self):
        '''
        contains the mu and sigma from pose
        '''
        self.mu_pose = np.eye(4) 
        self.sigma_pose = np.eye(6) 

    def predict(self, linear_velocity, rotational_velocity, tau, mu_current, sigma_current):
        '''
        EKF predict step and localization
        Input:
            linear_velocity: (3xt np.array) IMU linear velocity measurements in IMU frame
            rotational_velocity: (3xt np.array) IMU rotational velocity measurements in IMU frame
            tau: (float) delta tau timestep difference between t+1 and t
            mu_current: (4x4 np.array) pose of car in IMU frame
            sigma_current: (6x6 np.array) covariance of EKF
            
        Output:
            mu: (4x4 np.array) updated pose 
            sigma: (6x6 np.array) updated covariance
        '''

        vel = linear_velocity
        omega = rotational_velocity
        u = np.append(vel,omega) #6x1 vector with 2 3x1's [v w]

        #construct hat map of omega 
        omega_hatmap = np.array( [[0, -omega[2], omega[1]],
                                 [omega[2], 0, -omega[0]], 
                                 [-omega[1], omega[0], 0]] )
        #construct hat map of velocity 
        vel_hatmap = np.array( [[0, -vel[2], vel[1]],
                                 [vel[2], 0, -vel[0]], 
                                 [-vel[1], vel[0], 0]] )

        # uhat_t should be 4x4
        uhat_t = np.append( omega_hatmap, vel.reshape((3,1)), axis=1 )
        uhat_t = np.append( uhat_t, np.zeros((1,4)), axis=0 )
        #print("exponent is: ",expm( -tau * uhat_t ) )

        # ucurly_t should be 6x6
        u_top = np.append( omega_hatmap, vel_hatmap, axis=1)
        u_bot = np.append( np.zeros((3,3)), omega_hatmap, axis=1 )
        ucurly_t = np.append( u_top, u_bot, axis=0 )
        #print(ucurly_t)

        W = 0 #gaussian noise

        #create exponential map
        mu = expm( -tau * uhat_t ) @ mu_current
        sigma = expm( -tau * ucurly_t ) @ sigma_current @ np.transpose(expm( -tau * ucurly_t )) + W

        return mu, sigma



        

