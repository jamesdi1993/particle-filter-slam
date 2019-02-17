from utils.map_utils import recover_from_log_odds, xy_to_rc
from matplotlib.patches import Circle

import numpy as np
import math
import matplotlib.pyplot as plt


class Map():

  # def __init__(self, res, xmin, ymin, xmax, ymax):
  #   """
  #   :param xl: the length of the x-axis
  #   :param yl: the length of the y-axis
  #   :param x_resolution: resolution on the x-axis
  #   :param y_resolution: resolution on the y-axis;
  #   """
  #
  #   # Initialize map
  #   self.res = res # meters
  #   self.xmin = xmin  # meters
  #   self.ymin = ymin
  #   self.xmax = xmax
  #   self.ymax = ymax
  #   self.sizex = int(np.ceil((self.xmax - self.xmin) / self.res + 1))  # cells
  #   self.sizey = int(np.ceil((self.ymax - self.ymin) / self.res + 1))
  #   self.map = np.zeros((self.sizex, self.sizey), dtype=np.int8)  # DATA TYPE: char or int8

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
    pos = robot_pos.get_position()
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

  # TODO: Implement this method;
  def update_free(self, grids):
    """
    Update the log-likelihood of the map;
    :param grids: The grids that are observed to be free
    :return:
    """
    self.map[grids[0], grids[1]] = self.map[grids[0], grids[1]] + math.log(1 / self.error_ratio)

  # TODO: Implement this method;
  def update_occupied(self, grids):
    """
    Update the log-likelihood of the map;
    :param grids: The grids that are observed to be occupied
    :return: None
    """
    self.map[grids[0], grids[1]] = self.map[grids[0], grids[1]] + math.log(self.error_ratio)

  def check_range(self, x, y):
    return x < self.xmax and x > self.xmin and y < self.ymax and y > self.ymin

