from PIL import Image
import numpy as np

def load_data(dataset_index, load_image = False):
  dataset = {}

  with np.load("Encoders%d.npz"%dataset_index) as data:
    print("The keys of the encoders dataset_index are: %s" % (list(data.keys())))
    dataset['encoder_counts'] = data["counts"] # 4 x n encoder counts

    print("Printing the first 10 counts: %s" % (dataset['encoder_counts'][:4, :10]))
    dataset['encoder_stamps'] = data["time_stamps"] # encoder time stamps
    print("The shape of encoder_stamps is: %s" % (dataset['encoder_stamps'].shape, ))

  with np.load("Hokuyo%d.npz"%dataset_index) as data:
    print("The keys of the Hokuyo dataset_index are: %s" % (list(data.keys())))
    dataset['lidar_angle_min'] = data["angle_min"] # start angle of the scan [rad]

    print("The shape of lidar_angle_min is: %s" % (dataset['lidar_angle_min'].shape, ))

    dataset['lidar_angle_max'] = data["angle_max"] # end angle of the scan [rad]
    print("The shape of lidar_angle_max is: %s" % (dataset['lidar_angle_max'].shape,))

    dataset['lidar_angle_increment'] = data["angle_increment"] # angular distance between measurements [rad]
    print("The shape of lidar_angle_increment is: %s" % ( dataset['lidar_angle_increment'].shape,))
    print("The lidar_angle_increment is: %s" % (dataset['lidar_angle_increment'][0,0],))

    dataset['lidar_range_min'] = data["range_min"] # minimum range value [m]
    print("The shape of lidar_range_min is: %s" % (dataset['lidar_range_min'].shape,))

    dataset['lidar_range_max'] = data["range_max"] # maximum range value [m]
    print("The shape of lidar_range_max is: %s" % (dataset['lidar_range_max'].shape,))

    dataset['lidar_ranges'] = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded
    print("The shape of lidar_ranges is: %s" % (dataset['lidar_ranges'].shape,))

    dataset['lidar_stamps'] = data["time_stamps"]  # acquisition times of the lidar scans
    print("The shape of lidar_stamsp is: %s" % (dataset['lidar_stamps'].shape,))

    
  with np.load("Imu%d.npz"%dataset_index) as data:
    print("The keys of IMU dataset_index are: %s" % (list(data.keys())))
    dataset['imu_angular_velocity'] = data["angular_velocity"] # angular velocity in rad/sec
    print("The shape of imu_angular_velocity measurements is: %s" % (dataset['imu_angular_velocity'].shape, ))

    dataset['imu_linear_acceleration'] = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    print("The shape of imu_linear_acceleration measurements is: %s" % (dataset['imu_linear_acceleration'].shape, ))

    dataset['imu_stamps'] = data["time_stamps"]  # acquisition times of the imu measurements
    print("The shape of imu_stamps is: %s" % (dataset['imu_stamps'].shape,))

  if load_image:
    with np.load("Kinect%d.npz"%dataset_index) as data:
      print("The keys of Kinect data are: %s" % (list(data.keys())))
      dataset['disp_stamps'] = data["disparity_time_stamps"] # acquisition times of the disparity images
      print("The shape of disp_stamps is: %s" % (dataset['disp_stamps'].shape,))

      dataset['rgb_stamps'] = data["rgb_time_stamps"] # acquisition times of the rgb images
      print("The shape of rgb_stamps is: %s" % (dataset['rgb_stamps'].shape,))
  return dataset


def load_images(disp_index, correlated_timestamp, dataset_index):
  """
  Load rgb and disparitiy images.
  :param disp_index: The index of the disparity image.
  :param correlated_timestamp: The timestamps of the images.
  :return: disparity and rgb image.
  """
  rgb_index = correlated_timestamp[0, disp_index]

  depth_images_prefix = './dataRGBD/Disparity%s/disparity%s_' % (dataset_index, dataset_index)
  rgb_images_prefix = './dataRGBD/RGB%s/rgb%s_' % (dataset_index, dataset_index)

  depth_img_file = depth_images_prefix + str(correlated_timestamp[1, disp_index] + 1) + ".png"
  rgb_img_file = rgb_images_prefix + str(correlated_timestamp[0, rgb_index] + 1) + ".png"

  print("Reading depth image file: %s" % depth_img_file)
  print("Reading rgb image file: %s" % rgb_img_file)

  depth_img = Image.open(depth_img_file)
  disparity_img = np.array(depth_img.getdata(), np.uint16).reshape(depth_img.size[1], depth_img.size[0])

  rgb_img = Image.open(rgb_img_file)
  rgb_img = np.array(rgb_img.getdata(), np.uint8).reshape(rgb_img.size[1], rgb_img.size[0], 3)

  return disparity_img, rgb_img