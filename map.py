from utils.map_utils import recover_from_log_odds

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
    self.error_ratio = 4
    self.map = np.zeros((self.sizey, self.sizex), dtype=np.int8)  # DATA TYPE: char or int8

  def plot(self, epoch):
    """
    Plot the current map
    :return: N/A
    """
    # print("The maximum of log likelihoods is: %s" % np.max(self.map))
    # print("The minimum of log likelihoods is: %s" % np.min(self.map))
    map_prob = 1 - np.divide(np.ones(self.map.shape), 1 + np.exp(self.map))
    figure = plt.figure(figsize = (10,10))

    plt.imshow(map_prob, cmap="gray")
    plt.title('Displaying map at the %d epoch.' % epoch)
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

