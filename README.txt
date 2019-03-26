{\rtf1\ansi\ansicpg1252\cocoartf1404\cocoasubrtf470
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 This is the README file for the ECE276A Project 3 submission.\
\
4 Files are attached in this submission:\
\
EKF_SLAM.py\
EKF_predict.py\
EKF_update.py\
utils.py\
\
EKF_SLAM.py contains the main function used to do the EKF predict and update steps from parts a) and b) in the given tasks. EKF_predict contains the class used to predict the pose and covariance while EKF_update contains the update class that takes in input from the data and predicted pose from the prediction step. The utils.py is modified slightly in order to add the landmark scatter plots in the visualize_trajectory_2d function. Otherwise, the rest of utils.py is the same as the original file given from the starter code. \
\
\
\
Author: Erik Seetao}