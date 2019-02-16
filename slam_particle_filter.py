from load_data import load_data
from map import Map
from robot_pos import RobotPos
from utils.map_utils import transform_to_lidar_frame, transform_from_lidar_to_body_frame,\
  tranform_from_body_to_world_frame, bresenham2D, xy_to_rc
from utils.robot_utils import LIDAR_ANGLES, distance_per_tic
from utils.signal_process_utils import low_pass_imu

import math
import numpy as np
import matplotlib.pyplot as plt


def next_reading(encoder_counts, encoder_timestamps, imu_yaw_velocity, imu_timestamps, current_encoder_index,
                 current_imu_index):
  """
  Find the next encoder and imu yaw velocity reading.
  :param encoder_counts: The readings of the encoder
  :param encoder_timestamps: The timestamp of the encoder reading
  :param imu_yaw_velocity: The yaw velocity, recorded by IMU
  :param imu_timestamps: The IMU timestamp readings;
  :param current_index: The current index of the encoder timestamp
  :return: (last_encoder_index, last_imu_index, encoder_measurements average_yaw_velocity)
  """
  if current_encoder_index >= encoder_timestamps.shape[0] - 1:
    # Reach the end of the array
    # TODO: Revisit this logic;
    return (None, None, None, None)

  else:
    next_encoder_index = current_encoder_index + 1
    next_encoder_timestamp = encoder_timestamps[next_encoder_index]

    # print("The shape of encoder_counts is: %s" % (encoder_counts.shape, ))
    # print("The shape of encoder index is: %s" % (next_encoder_index,))
    next_encoder_count = encoder_counts[:, next_encoder_index]

    yaw_velocity = []
    next_imu_index = current_imu_index
    for i in range(current_imu_index + 1, imu_timestamps.shape[0]):
      # If there are imu readings between the two encoder measurements;
      if imu_timestamps[i] < next_encoder_timestamp:
        yaw_velocity.append(imu_yaw_velocity[i])
        next_imu_index = i
      else:
        break
    yaw_v_average = 0
    if len(yaw_velocity) != 0:
      yaw_v_average = sum(yaw_velocity) / len(yaw_velocity)
    return (next_encoder_index, next_imu_index, next_encoder_count, yaw_v_average)



if __name__ == '__main__':
  # Load data
  data = load_data(dataset_index = 20)
  config = {
    'res': 0.5,
    'xmin': -50,
    'xmax': 50,
    'ymin': -50,
    'ymax': 50
  }
  
  # Initialize map;
  map = Map(config)
  map.plot(epoch = 0)


  # Initalize robot position
  robot_pos = RobotPos(numParticles= 1)

  # Plotting the trajectory of the robot;
  encoder_counts = data['encoder_counts']
  encoder_timestamp = data['encoder_stamps']

  imu_yaw_velocity = data['imu_angular_velocity'][2, :]
  filtered_imu_yaw_velocity = low_pass_imu(imu_yaw_velocity)

  # figure = plt.figure(figsize=(10,10))
  # plt.plot(imu_yaw_velocity, 'b')
  # plt.plot(filtered_imu_yaw_velocity, 'r')
  # plt.show()
  #
  # print("The first five yaw velocity are: %s" % imu_yaw_velocity[0:5])

  # Low pass filter with threshold = 0.5
  imu_timestamps = data['imu_stamps']

  current_imu_index = -1
  current_encoder_index = -1

  current_encoder_index, current_imu_index, current_encoder_counts, yaw_v_average = next_reading(encoder_counts,
                                                                                       encoder_timestamp,
                                                                                       filtered_imu_yaw_velocity,
                                                                                       imu_timestamps,
                                                                                       current_encoder_index,
                                                                                       current_imu_index)

  # Initializing robot trajectory; n + 1 timestamps because we initialize trajectory to be [0,0,0] at t = -1
  robot_trajectory = np.zeros((3, encoder_timestamp.shape[0]))


  while (current_encoder_index is not None):
    time_elapsed = 0.025 if current_encoder_index == 0 else \
      encoder_timestamp[current_encoder_index] - encoder_timestamp[current_encoder_index - 1]
    d_w = time_elapsed * yaw_v_average
    d_r = (current_encoder_counts[0] + current_encoder_counts[2]) * distance_per_tic / 2  # (FR + RR) * 0.0022 / 2
    d_l = (current_encoder_counts[1] + current_encoder_counts[3]) * distance_per_tic / 2  # (FL + RL) * 0.0022 / 2
    d = (d_r + d_l) / 2

    x = 0
    y = 0
    theta = 0

    if (current_encoder_index == 0):
      theta = d_w
      x = d * math.cos(theta)
      y = d * math.sin(theta)
    else:
      theta = (robot_trajectory[2, current_encoder_index - 1] + d_w) % (2 * math.pi)
      x = robot_trajectory[0, current_encoder_index - 1] + d * math.cos(theta)
      y = robot_trajectory[1, current_encoder_index - 1] + d * math.sin(theta)

    # Update robot trajectory
    robot_trajectory[:, current_encoder_index] = np.array([x, y, theta])
    current_encoder_index, current_imu_index, current_encoder_counts, yaw_v_average = next_reading(encoder_counts,
                                                                                           encoder_timestamp,
                                                                                           filtered_imu_yaw_velocity,
                                                                                           imu_timestamps,
                                                                                           current_encoder_index,
                                                                                           current_imu_index)

  print("The shape of robot_trajectory is: %s" % (robot_trajectory.shape, ))
  trajectory_coordinate = xy_to_rc(map.xrange, map.yrange, robot_trajectory[0, :], robot_trajectory[1, :], map.res)
  map.plot_robot_trajectory(trajectory_coordinate)

  print("The shape of trajectory_coordinate is: %s" % (trajectory_coordinate.shape,))
  print("The first five coordinates of the robot are: %s" % trajectory_coordinate[:, :5])
  print("The last five coordinates of the robot are: %s" % trajectory_coordinate[:, -5:])
  print("Finished plotting the data.")
  # print("The shape of the yaw velocities are: %s" % (imu_yaw_velocity.shape, ))
  #
  # print("The first five yaw velocities are: %s" % (imu_yaw_velocity[:5],))
  # print("The first five timestamps from IMU are: %s" % (imu_timestamps[:5],))
  #
  #
  # imu_diff = np.sum((imu_timestamps[1:] - imu_timestamps[:-1]))/ (imu_timestamps.shape[0] - 1)
  # print("The average difference between two IMU measurements is: %s" % imu_diff)
  # print("The max and min in IMU measurements is: %s, %s" % (np.max(imu_yaw_velocity), np.min(imu_yaw_velocity)))
  #
  #
  # encoder_diff =  np.sum((encoder_timestamp[1:] - encoder_timestamp[:-1]))/ (encoder_timestamp.shape[0] - 1)
  # print("The average difference between two encoder measurements is: %s" % encoder_diff)
  #
  # print("The first five encoder counts are: %s" % (encoder_counts[:, :5],))
  # print("The first five timestamps for encoder are: %s" % (encoder_timestamp[:5],))

