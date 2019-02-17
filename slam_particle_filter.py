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

def get_last_lidar_measurement(encoder_timestamp, current_lidar_index, lidar_timestamps):
  """
  Based on the current time stamp of the encoder, give the last measurement of the lidar;
  :param encoder_timestamp: The timestamp of the encoder;
  :param current_lidar_index: The index of the previous lidar measurement;
  :param lidar_timestamps: The timestamps of the lidar measurements;
  :return: The index of the next lidar measurment;
  """
  next_measurement_index = current_lidar_index
  for i in range(current_lidar_index + 1, lidar_timestamps.shape[0]):
    if lidar_timestamps[i] > encoder_timestamp:
      # return previous lidar index
      break
    else:
      next_measurement_index = i
  return next_measurement_index




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


  # Initalize robot position
  robot_pos = RobotPos(numParticles= 1)
  
  # Initialize map;
  map = Map(config)
  map.plot(robot_pos, epoch = 0)


  # Plotting the trajectory of the robot;
  encoder_counts = data['encoder_counts']
  encoder_timestamp = data['encoder_stamps']

  imu_yaw_velocity = data['imu_angular_velocity'][2, :]
  filtered_imu_yaw_velocity = low_pass_imu(imu_yaw_velocity)

  # Transform the lidar data to the body frame;
  lidar_timestamps = data['lidar_stamps']
  lidar_range = data['lidar_ranges']

  lidar_xy = transform_to_lidar_frame(lidar_range, LIDAR_ANGLES)
  lidar_body = transform_from_lidar_to_body_frame(lidar_xy)

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

  # Start with the zero index;
  current_lidar_index = get_last_lidar_measurement(encoder_timestamp[current_encoder_index], 0, lidar_timestamps)

  current_lidar_body = lidar_body[:, current_lidar_index, :]

  # Initializing robot trajectory; n + 1 timestamps because we initialize trajectory to be [0,0,0] at t = -1
  robot_trajectory = np.zeros((3, encoder_timestamp.shape[0]))

  """
  Main loop for running the following steps:
  1) Update the map based on current guess of the robot;
  2) Update the position for each particle based on the robot motion model; 
  3) Update the particle weights based on observation;
  4) Re-sample the particle; 
  """
  while (current_encoder_index is not None):
    # Mapping;
    print("Executing for the %dth epoch." % (current_encoder_index,))
    position = robot_pos.get_position()
    current_lidar_world = tranform_from_body_to_world_frame(position, current_lidar_body)
    print("The shape of the lidar rays in world frame is: %s" % (current_lidar_world.shape,))

    # Plot obstacles for the current frame;
    for i in range(current_lidar_world.shape[0]):

      x = current_lidar_world[i, 0]
      y = current_lidar_world[i, 1]

      if not math.isnan(x) and not math.isnan(y) and map.check_range(x, y):
        # Update free cells;
        grids_xy = bresenham2D(position[0, 0], position[1, 0], x, y)
        grids_rc = xy_to_rc(map.xrange, map.yrange, grids_xy[0], grids_xy[1], map.res)
        map.update_free(grids_rc)

        # Update occupied cells;
        end_rc = xy_to_rc(map.xrange, map.yrange, np.array([x]), np.array([y]), map.res)
        map.update_occupied(end_rc)


    # Plot every 200 steps:
    if current_encoder_index % 200 == 0:
      map.plot(robot_pos, epoch=current_encoder_index)

    # Prediction;
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

    # TODO: Add observation step here.

    # Update robot position
    pos_new = np.array([x, y, theta])
    robot_pos.update_position(np.expand_dims(pos_new, axis = 1))
    robot_trajectory[:, current_encoder_index] = pos_new


    current_encoder_index, current_imu_index, current_encoder_counts, yaw_v_average = next_reading(encoder_counts,
                                                                                           encoder_timestamp,
                                                                                           filtered_imu_yaw_velocity,
                                                                                           imu_timestamps,
                                                                                           current_encoder_index,
                                                                                           current_imu_index)
    # Get the next lidar measurement;
    current_lidar_index = get_last_lidar_measurement(encoder_timestamp[current_encoder_index], current_lidar_index, lidar_timestamps)
    current_lidar_body = lidar_body[:, current_lidar_index, :]


  """
  Plot robot trajectory 

  print("The shape of robot_trajectory is: %s" % (robot_trajectory.shape, ))
  trajectory_coordinate = xy_to_rc(map.xrange, map.yrange, robot_trajectory[0, :], robot_trajectory[1, :], map.res)
  map.plot_robot_trajectory(trajectory_coordinate)

  print("The shape of trajectory_coordinate is: %s" % (trajectory_coordinate.shape,))
  print("The first five coordinates of the robot are: %s" % trajectory_coordinate[:, :5])
  print("The last five coordinates of the robot are: %s" % trajectory_coordinate[:, -5:])
  print("Finished plotting the data.")
  """



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

