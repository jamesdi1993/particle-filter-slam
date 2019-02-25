from load_data import load_data, load_images
from map import Map
from robot_pos import RobotPos
from utils.map_utils import transform_to_lidar_frame, transform_from_lidar_to_body_frame,\
  tranform_from_body_to_world_frame, xy_to_rc, tic, toc, correlate_timestamp, transform_from_d
from utils.robot_utils import LIDAR_ANGLES, LIDAR_MAX, LIDAR_MIN_JITTER
from utils.signal_process_utils import low_pass_imu
from utils.camera_utils import from_homog, to_homogenuous, transform_from_camera_to_body_frame

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

def get_last_observation(encoder_timestamp, current_observation_index, observation_timestamps):
  """
  Based on the current time stamp of the encoder, give the last measurement of the lidar;
  :param encoder_timestamp: The timestamp of the encoder;
  :param current_observation_index: The index of the previous lidar measurement;
  :param observation_timestamps: The timestamps of the lidar measurements;
  :return: The index of the next lidar measurment;
  """
  next_observation_index = current_observation_index
  for i in range(current_observation_index + 1, observation_timestamps.shape[0]):
    if observation_timestamps[i] > encoder_timestamp:
      # return previous lidar index
      break
    else:
      next_observation_index = i
  return next_observation_index

def process_image(rgb_image, disparity_img, robot_pos, map):
  # start = tic()
  rgbi, rgbj, depth = transform_from_d(disparity_img)
  valid_pixels = np.logical_and(rgbi >= 0, rgbi < 640)
  valid_pixels = np.logical_and.reduce((valid_pixels, rgbj >= 0, rgbj < 480, depth > 0))

  valid_depth = np.expand_dims(depth[valid_pixels], axis=1)
  valid_rgbj = np.expand_dims(rgbj[valid_pixels], axis=1)
  valid_rgbi = np.expand_dims(rgbi[valid_pixels], axis=1)

  rgb_ijd = np.concatenate((valid_rgbi, valid_rgbj, valid_depth), axis=1)

  ground_positions = np.zeros((rgb_ijd.shape[0], 2))
  ground_count = 0
  rgb_values = np.zeros((rgb_ijd.shape[0], 3))
  for i in range(rgb_ijd.shape[0]):
    # print("The pixels coordinate is: %s; Depth is: %s" % (rgb_jid[i, 0:2],rgb_jid[i, -1]))
    p = from_homog(transform_from_camera_to_body_frame(np.expand_dims(rgb_ijd[i, :], axis=1)))
    # z_dist[i] = p[-1]
    # print("The z of the pixel in the robot frame is: %s" % p[-1])

    # mask out pixels that are below or over a threshold.
    if p[-1] < 0:
      ground_positions[ground_count, :] = p[:-1].reshape(2)
      rgb_values[ground_count, :] = rgb_img[int(rgb_ijd[i, 1]), int(rgb_ijd[i, 0]), :]
      ground_count += 1

  ground_positions = ground_positions[:ground_count, :]
  rgb_values = rgb_values[:ground_count, :]
  world_pos = tranform_from_body_to_world_frame(robot_pos, ground_positions) #n x 2
  rc = xy_to_rc(map.xmin, map.ymin, world_pos[:, 0], world_pos[:, 1], map.res)

  for i in range(rc.shape[1]):
    map.update_texture(rc[:, i], rgb_values[i, :])
  # toc(start)

if __name__ == '__main__':
  d_sigma = float(sys.argv[1])
  yaw_sigma = float(sys.argv[2])
  num_particles = int(sys.argv[3])
  save_fig = bool(int(sys.argv[4]))
  texture_map = bool(int(sys.argv[5]))
  dataset_index = int(sys.argv[6])

  directory = "final-report/"
  # if num_particles == 1:
  #   directory = "dead-reckoning"
  # else:
  directory = directory + str(dataset_index) + "/" + str(d_sigma) + "-" + str(yaw_sigma) + "-" + str(num_particles)
  os.makedirs(directory)

  print("The linear velocity noise is: %s" % d_sigma)
  print("The yaw velocity noise is: %s" % yaw_sigma)
  print("The number of particles: %s" % num_particles)

  start_time = tic()

  # Load data
  data = load_data(dataset_index = dataset_index, load_image = texture_map)
  config = {
    'res': 0.05,
    'xmin': -10,
    'xmax': 30,
    'ymin': -10,
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
  encoder_counts = data['encoder_counts'][:, 400:-1] # Skipping the first 400 hundred measurements;
  encoder_timestamp = data['encoder_stamps'][400:-1]

  imu_yaw_velocity = data['imu_angular_velocity'][2, :]
  imu_timestamps = data['imu_stamps']

  filtered_imu_yaw_velocity = low_pass_imu(imu_yaw_velocity)

  # Transform the lidar data to the body frame;
  lidar_timestamps = data['lidar_stamps']
  lidar_range = data['lidar_ranges'] # 1081 x 4962

  # Image manipulation
  depth_images_prefix = './dataRGBD/Disparity%s/disparity%s_' % (dataset_index, dataset_index)
  rgb_images_prefix = './dataRGBD/RGB%s/rgb%s_' % (dataset_index, dataset_index)

  disp_timestamp = data.get('disp_stamps')
  rgb_timestamp = data.get('rgb_stamps')
  image_timestamp = None
  if disp_timestamp is not None and rgb_timestamp is not None:
    image_timestamp = correlate_timestamp(rgb_timestamp, disp_timestamp)

  current_imu_index = -1
  current_encoder_index = -1
  current_disp_index = -1
  current_lidar_index = -1

  current_encoder_index, current_imu_index, current_encoder_counts, yaw_v_average = next_reading(encoder_counts,
                                                                                                 encoder_timestamp,
                                                                                                 filtered_imu_yaw_velocity,
                                                                                                 imu_timestamps,
                                                                                                 current_encoder_index,
                                                                                                 current_imu_index)
  current_lidar_index = get_last_observation(encoder_timestamp[current_encoder_index], current_lidar_index,
                                             lidar_timestamps)
  current_lidar = lidar_range[:, current_lidar_index]
  current_lidar_xy = transform_to_lidar_frame(current_lidar, LIDAR_ANGLES, LIDAR_MIN_JITTER + map.res, LIDAR_MAX)
  current_lidar_body = transform_from_lidar_to_body_frame(current_lidar_xy)
  current_lidar_world = tranform_from_body_to_world_frame(robot_pos.get_best_particle_pos(), current_lidar_body)


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
    position = np.copy(robot_pos.get_best_particle_pos())
    # Plot obstacles for the current frame; Update log odds;
    map.update_log_odds(current_lidar_world, position)
    robot_trajectory[:, index] = position
    index += 1

    current_encoder_index, current_imu_index, current_encoder_counts, yaw_v_average = next_reading(encoder_counts,
                                                                                                   encoder_timestamp,
                                                                                                   filtered_imu_yaw_velocity,
                                                                                                   imu_timestamps,
                                                                                                   current_encoder_index,
                                                                                                   current_imu_index)
    # Prediction;
    if current_encoder_index is not None:
      time_elapsed = 0.025 if current_encoder_index == 0 else \
        encoder_timestamp[current_encoder_index] - encoder_timestamp[current_encoder_index - 1]

      # Predict the position of the particles;
      robot_pos.predict_particles(current_encoder_counts, yaw_v_average, time_elapsed, d_sigma, yaw_sigma)
      # robot_pos.predict_particles(current_encoder_counts, yaw_v_average, time_elapsed)

      # if current_encoder_index % 1000 == 0:
      #   title = "Prediction step at the %dth epoch. D_sigma: %s; Yaw_sigma: %s; Num_particles: %s" % \
      #           (current_encoder_index, d_sigma, yaw_sigma, num_particles)
      #   map.plot(robot_pos = robot_pos, robot_trajectory=None, title=title,
      #            img_name= directory + '/' + str(current_encoder_index) + '-' + "Predict",
      #            save_fig=save_fig)

        # Get the next lidar measurement;
      current_lidar_index = get_last_observation(encoder_timestamp[current_encoder_index], current_lidar_index,
                                                 lidar_timestamps)
      current_lidar = lidar_range[:, current_lidar_index]
      current_lidar_xy = transform_to_lidar_frame(current_lidar, LIDAR_ANGLES, LIDAR_MIN_JITTER + map.res, LIDAR_MAX)
      current_lidar_body = transform_from_lidar_to_body_frame(current_lidar_xy)

      # Image reading
      if current_encoder_index % 5 == 0 and texture_map:
        current_disp_index = get_last_observation(encoder_timestamp[current_encoder_index], 0,
                                                  disp_timestamp[image_timestamp[1, :]])
        disp_img, rgb_img = load_images(current_disp_index, image_timestamp, dataset_index)

      current_lidar_world = tranform_from_body_to_world_frame(position, current_lidar_body)

      # Update the positions of the particles
      if num_particles != 1:
        # update posterior if number of particles is greater than 1;
        robot_pos.update_particles(current_lidar_body, map, deviation=linear_deviation, yaw_deviation=yaw_deviation)

      # Plot every 100 steps for particles update
      if current_encoder_index % 1000 == 0:
        title = "Update step at the %dth epoch. D_sigma: %s; Yaw_sigma: %s; Num_particles: %s;" % \
                (current_encoder_index, d_sigma, yaw_sigma, num_particles)
        if texture_map:
          map.plot(robot_pos=None, robot_trajectory=robot_trajectory, title=title,
                   img_name=directory + '/' + str(current_encoder_index) + '-' + "Update",
                   save_fig=save_fig)
        else:
          map.plot(robot_pos = robot_pos, robot_trajectory=None, title=title,
                   img_name=directory + '/' + str(current_encoder_index) + '-' + "Update",
                   save_fig=save_fig)

      # Texture map the floor
      if current_encoder_index % 5 == 0 and texture_map:
        # Plot every half second
        process_image(rgb_img, disp_img, position, map)
        title = "Texture map at %sth epoch" % current_encoder_index
        map.plot_texture(title, img_name = directory + "/" + "Texture_mapping-" + str(current_encoder_index),
                         save_fig = save_fig)

  title = "Finish map. D_sigma: %s; Yaw_sigma: %s; Num_particles: %s;" % \
          (d_sigma, yaw_sigma, num_particles)
  map.plot(robot_pos = None, robot_trajectory=robot_trajectory, title=title,
           img_name=directory + '/' + str(current_encoder_index) + '-' + "Final",
           save_fig=save_fig)

  toc(start_time)
  print("Finished plotting the data.")

