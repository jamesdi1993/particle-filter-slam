from load_data import load_data
from map import Map
from robot_pos import RobotPos
from utils.map_utils import transform_to_lidar_frame, transform_from_lidar_to_body_frame,\
  tranform_from_body_to_world_frame, xy_to_rc, tic, toc
from utils.robot_utils import LIDAR_ANGLES, LIDAR_MAX, LIDAR_MIN_JITTER
from utils.signal_process_utils import low_pass_imu

import numpy as np
import os
import sys


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
  d_sigma = float(sys.argv[1])
  yaw_sigma = float(sys.argv[2])
  num_particles = int(sys.argv[3])
  save_fig = bool(sys.argv[4])

  directory = str(d_sigma) + '-' + str(yaw_sigma) + '-' + str(num_particles)
  os.mkdir(directory)

  print("The linear velocity noise is: %s" % d_sigma)
  print("The yaw velocity noise is: %s" % yaw_sigma)
  print("The number of particles: %s" % num_particles)

  start_time = tic()

  # Load data
  data = load_data(dataset_index = 20)
  config = {
    'res': 0.2,
    'xmin': -30,
    'xmax': 30,
    'ymin': -30,
    'ymax': 30
  }

  # d_sigma = 0.1 # Controls the spread of the particles
  # yaw_sigma = 0.2 / 0.025  # deviation for yaw, devide by 0.025 since the time duration is 0.025.
  linear_deviation = 0.4
  yaw_deviation = 0.1

  # Initalize robot position
  initial_pos = np.array([0,0,0]) # x, y, theta
  robot_pos = RobotPos(initial_pos, numParticles=num_particles, particle_threshold= 0.5 * num_particles)
  
  # Initialize map;
  map = Map(config)
  title = "Displaying robot at the 0 epoch."
  map.plot(robot_pos, robot_trajectory = None, title = title, save_fig = False)


  # Preprocess IMU and Encoder measurements;
  encoder_counts = data['encoder_counts'][:, 300:] # Skipping the first 300 hundred measurements;
  encoder_timestamp = data['encoder_stamps'][300:]

  imu_yaw_velocity = data['imu_angular_velocity'][2, :]
  imu_timestamps = data['imu_stamps']

  filtered_imu_yaw_velocity = low_pass_imu(imu_yaw_velocity)

  # Transform the lidar data to the body frame;
  lidar_timestamps = data['lidar_stamps']
  lidar_range = data['lidar_ranges'] # 1081 x 4962

  """
  figure = plt.figure(figsize=(10,10))
  plt.plot(imu_yaw_velocity, 'b')
  plt.plot(filtered_imu_yaw_velocity, 'r')
  plt.show()
  
  print("The first five yaw velocity are: %s" % imu_yaw_velocity[0:5])

  Low pass filter with threshold = 0.5
  """

  current_imu_index = -1
  current_encoder_index = -1

  # First reading from IMU and Encoder;
  current_encoder_index, current_imu_index, current_encoder_counts, yaw_v_average = next_reading(encoder_counts,
                                                                                       encoder_timestamp,
                                                                                       filtered_imu_yaw_velocity,
                                                                                       imu_timestamps,
                                                                                       current_encoder_index,
                                                                                       current_imu_index)

  # First lidar reading;
  current_lidar_index = get_last_lidar_measurement(encoder_timestamp[current_encoder_index], 0, lidar_timestamps)
  current_lidar = lidar_range[:, current_lidar_index]
  current_lidar_xy = transform_to_lidar_frame(current_lidar, LIDAR_ANGLES, LIDAR_MIN_JITTER + map.res, LIDAR_MAX) # 1081 x 2
  current_lidar_body = transform_from_lidar_to_body_frame(current_lidar_xy)

  robot_trajectory = np.zeros((3, encoder_counts.shape[1]))
  index = 0
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
    position = robot_pos.get_best_particle_pos()
    robot_trajectory[:, index] = position
    index += 1

    current_lidar_world = tranform_from_body_to_world_frame(position, current_lidar_body)

    # Plot obstacles for the current frame; Update log odds;
    map.update_log_odds(current_lidar_world, position)

    # Prediction;
    time_elapsed = 0.025 if current_encoder_index == 0 else \
      encoder_timestamp[current_encoder_index] - encoder_timestamp[current_encoder_index - 1]

    # Predict the position of the particles;
    robot_pos.predict_particles(current_encoder_counts, yaw_v_average, time_elapsed, d_sigma, yaw_sigma)
    # robot_pos.predict_particles(current_encoder_counts, yaw_v_average, time_elapsed)

    if current_encoder_index is not None and current_encoder_index % 1000 == 0:
      title = "After prediction step displaying map at the %dth epoch. D_sigma: %s; Yaw_sigma: %s; Num_particles: %s" % \
              (current_encoder_index, d_sigma, yaw_sigma, num_particles)
      map.plot(robot_pos = None, robot_trajectory=robot_trajectory, title=title,
               img_name= directory + '/' + str(current_encoder_index) + '-' + "Predict",
               save_fig=save_fig)

    # Update the positions of the particles
    robot_pos.update_particles(current_lidar_body, map, deviation=linear_deviation, yaw_deviation=yaw_deviation)

    # Get the next encoder measurements;

    current_encoder_index, current_imu_index, current_encoder_counts, yaw_v_average = next_reading(encoder_counts,
                                                                                           encoder_timestamp,
                                                                                           filtered_imu_yaw_velocity,
                                                                                           imu_timestamps,
                                                                                           current_encoder_index,
                                                                                           current_imu_index)
    if current_encoder_index is not None:
      # Get the next lidar measurement;
      current_lidar_index = get_last_lidar_measurement(encoder_timestamp[current_encoder_index], current_lidar_index, lidar_timestamps)
      current_lidar = lidar_range[:, current_lidar_index]
      current_lidar_xy = transform_to_lidar_frame(current_lidar, LIDAR_ANGLES, LIDAR_MIN_JITTER + map.res, LIDAR_MAX)
      current_lidar_body = transform_from_lidar_to_body_frame(current_lidar_xy)

    # Plot every 1000 steps:
    if current_encoder_index is not None and current_encoder_index % 1000 == 0:
      title = "After update step displaying map at the %dth epoch. D_sigma: %s; Yaw_sigma: %s; Num_particles: %s;" % \
              (current_encoder_index, d_sigma, yaw_sigma, num_particles)
      map.plot(robot_pos = None, robot_trajectory=robot_trajectory, title=title,
               img_name=directory + '/' + str(current_encoder_index) + '-' + "Update",
               save_fig=save_fig)

  # print("The shape of robot_trajectory is: %s" % (robot_trajectory.shape, ))
  # trajectory_coordinate = xy_to_rc(map.xmin, map.ymin, robot_trajectory[0, :], robot_trajectory[1, :], map.res)
  # map.plot_robot_trajectory(trajectory_coordinate)
  title = "Finish map. D_sigma: %s; Yaw_sigma: %s; Num_particles: %s;" % \
          (d_sigma, yaw_sigma, num_particles)
  map.plot(robot_pos = None, robot_trajectory=robot_trajectory, title=title,
           img_name=directory + '/' + str(current_encoder_index) + '-' + "Final",
           save_fig=save_fig)

  toc(start_time)
  print("Finished plotting the data.")

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

