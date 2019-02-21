# util function and hardware specifications for the robot;

import numpy as np

LIDAR_ANGLES = np.arange(-135,135.25, 0.25) * np.pi/180.0
LIDAR_MIN_JITTER = 0.2
LIDAR_MAX = 30

P_LIDAR_TO_BODY = [0.29833, 0, 0.51435]

distance_per_tic = 0.0022
yaw_deviation_res = 0.02 # 1.14 degrees

