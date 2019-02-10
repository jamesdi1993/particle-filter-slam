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
    self.sizex = int(np.ceil((self.xmax - self.xmin) / self.res + 1))  # cells
    self.sizey = int(np.ceil((self.ymax - self.ymin) / self.res + 1))
    self.map = np.zeros((self.sizey, self.sizex), dtype=np.int8)  # DATA TYPE: char or int8

  def plot(self, epoch):
    """
    Plot the current map
    :return: N/A
    """
    figure = plt.figure(figsize = (10,10))
    plt.imshow(self.map, cmap="hot")
    plt.title('Displaying map at the %d epoch.' % epoch)
    plt.show()


  def update(self):
    """
    Update the map according to current measurements;
    :return:
    """
    pass
