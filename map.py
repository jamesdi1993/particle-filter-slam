from utils.map_utils import bresenham2D, recover_from_log_odds, xy_to_rc
from matplotlib.patches import Circle

import numpy as np
import math
import matplotlib.pyplot as plt


class Map():

  def __init__(self, *args, **kargs):
    self.initializeFromConfig(args[0])

  def initializeFromConfig(self, config):
    self.res = config['res']
    self.xmin = config['xmin']
    self.xmax = config['xmax']
    self.ymin = config['ymin']
    self.ymax = config['ymax']
    if not ((self.xmax - self.xmin) / self.res).is_integer():
      raise ValueError("xmax - xmin must be divided evenly by res. xmax: %s, xmin: %s, res: %s"
                                 % (self.xmax, self.xmin, self.res))
    if not ((self.ymax - self.ymin) / self.res).is_integer():
      raise ValueError("ymax - ymin must be divided evenly by res. xmax: %s, xmin: %s, res: %s"
                                 % (self.ymax, self.ymin, self.res))
    self.sizex = int((self.xmax - self.xmin) / self.res + 1)  # cells
    self.sizey = int((self.ymax - self.ymin) / self.res + 1)
    self.xrange = self.xmax - self.xmin
    self.yrange = self.ymax - self.ymin
    self.error_ratio = 4
    self.map = np.zeros((self.sizey, self.sizex), dtype=np.int8)  # DATA TYPE: char or int8

  def plot(self, robot_pos, epoch):
    """
    Plot the current map, and the robot pos
    :return: N/A
    """
    # print("The maximum of log likelihoods is: %s" % np.max(self.map))
    # print("The minimum of log likelihoods is: %s" % np.min(self.map))
    map_prob = 1 - np.divide(np.ones(self.map.shape), 1 + np.exp(self.map))
    pos = robot_pos.get_best_particle_pos()
    pos_rc = xy_to_rc(self.xrange, self.yrange, pos[0], pos[1], self.res)
    circ = Circle((pos_rc[0], pos_rc[1]), 1, color='blue')
    figure, ax = plt.subplots(1)

    ax.imshow(map_prob, cmap="gray")
    ax.add_patch(circ)
    plt.title('Displaying map at the %d epoch.' % epoch)

    plt.show()

  def plot_robot_trajectory(self, trajectory):
    print("The max index of trajectory is: %s" % np.max(trajectory))
    figure = plt.figure(figsize=(10, 10))
    trajectory_map = np.zeros(self.map.shape)
    trajectory_map[trajectory[0], trajectory[1]] = 1
    plt.imshow(trajectory_map, cmap="hot")
    plt.title('Displaying robot trajectory')
    plt.show()


  def update_free(self, grids):
    """
    Update the log-likelihood of the map;
    :param grids: The grids that are observed to be free
    :return:
    """
    self.map[grids[0], grids[1]] = self.map[grids[0], grids[1]] + math.log(1 / self.error_ratio)

  def update_occupied(self, grids):
    """
    Update the log-likelihood of the map;
    :param grids: The grids that are observed to be occupied
    :return: None
    """
    self.map[grids[0], grids[1]] = self.map[grids[0], grids[1]] + math.log(self.error_ratio)

  def check_range(self, x, y):
    return x < self.xmax and x > self.xmin and y < self.ymax and y > self.ymin


  def update_log_odds(self, current_lidar_world, robot_pos):
    """
    Update the log_odds of the map
    :param current_lidar_world: The lidar readings in the most recent frame.
    :param robot_pos: The position of the robot;
    :return:
    """
    # Plot obstacles for the current frame;
    for i in range(current_lidar_world.shape[0]):

      x = current_lidar_world[i, 0]
      y = current_lidar_world[i, 1]

      if not math.isnan(x) and not math.isnan(y) and self.check_range(x, y):
        # Update free cells;
        grids_xy = bresenham2D(robot_pos[0], robot_pos[1], x, y)
        grids_rc = xy_to_rc(self.xrange, self.yrange, grids_xy[0], grids_xy[1], self.res)
        self.update_free(grids_rc)

        # Update occupied cells;
        end_rc = xy_to_rc(self.xrange, self.yrange, np.array([x]), np.array([y]), self.res)
        self.update_occupied(end_rc)

