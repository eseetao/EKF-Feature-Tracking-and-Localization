#EKF Feature Tracking and Localization

4 Files are attached in this project:

EKF_SLAM.py
EKF_predict.py
EKF_update.py
utils.py

EKF_SLAM.py contains the main function used to perform the EKF predict and update steps. EKF_predict contains the class used to predict the pose and covariance while EKF_update contains the update class that takes in input from the data and predicted pose from the prediction step. The utils.py is modified slightly in order to add the landmark scatter plots in the visualize_trajectory_2d function. Otherwise, the rest of utils.py is the same as the original file given from the starter code. 



Author: Erik Seetao
