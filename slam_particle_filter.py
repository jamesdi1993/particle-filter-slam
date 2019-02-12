from load_data import load_data
from map import Map
from utils.map_utils import transform_to_lidar_frame, transform_from_lidar_to_body_frame,\
  tranform_from_body_to_world_frame, bresenham2D, xy_to_rc
from utils.robot_utils import LIDAR_ANGLES

import math
import numpy as np

if __name__ == '__main__':
  # Load data
  data = load_data(dataset_index = 20)
  config = {
    'res': 0.25,
    'xmin': -20,
    'xmax': 20,
    'ymin': -20,
    'ymax': 20
  }
  
  # Initialize map;
  map = Map(config)
  map.plot(epoch = 0)

  # Transform the lidar data to the body frame;
  lidar_data = data['lidar_ranges']
  lidar_xy = transform_to_lidar_frame(lidar_data, LIDAR_ANGLES)
  lidar_body = transform_from_lidar_to_body_frame(lidar_xy)

  # return position of the robot;
  position = np.array([[0],[0],[0]])
  lidar_world = tranform_from_body_to_world_frame(position, lidar_body)
  print("The shape of the lidar rays in world frame is: %s" % (lidar_world.shape,))
  print("The first five coordinates in the world frame is: %s" % (lidar_world[:5, 0, :]))

  start_frame = lidar_world[:, 0, :]

  # Plot obstacles from first laser scan
  for i in range(start_frame.shape[0]):

    x = start_frame[i, 0]
    y = start_frame[i, 1]

    if not math.isnan(x) and not math.isnan(y):
      # Update free cells;
      grids_xy = bresenham2D(position[0, 0], position[1, 0], x, y)
      grids_rc = xy_to_rc(map.sizex, map.sizey, grids_xy[0], grids_xy[1])
      map.update_free(grids_rc)

      # Update occupied cells;
      end_rc = xy_to_rc(map.sizex, map.sizey, np.array([x]), np.array([y]))
      map.update_occupied(end_rc)

  map.plot(epoch = 1)
  print("Finished plotting for the first epoch.")

