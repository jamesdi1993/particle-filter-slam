from utils.map_utils import bresenham2D, recover_from_log_odds, xy_to_rc
from utils.robot_utils import LIDAR_ANGLES
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
    self.map = np.zeros((self.sizex, self.sizey), dtype=np.float64)  # DATA TYPE: char or int8
    self.log_odds_min = -10   # log-odds min ratio
    self.log_odds_max = 10    # log-odds max ratio


  def get_binary_map(self):
    """
    Get the binary version of the log odds map
    :return: the binary log map;
    """
    binary_map = (self.map > 0).astype(int)
    return binary_map


  def plot(self, robot_pos, title, img_name=None, save_fig=False):
    """
    Plot the current map, and the robot pos
    :return: N/A
    """
    # print("The maximum of log likelihoods is: %s" % np.max(self.map))
    # print("The minimum of log likelihoods is: %s" % np.min(self.map))
    map_prob = 1 - np.divide(np.ones(self.map.shape), 1 + np.exp(self.map))
    pos = robot_pos.get_best_particle_pos()
    particle_positions = robot_pos.get_particles_pos() # 3 x n array

    pos_rc = xy_to_rc(self.xmin, self.ymin, pos[0], pos[1], self.res)
    circ = Circle((pos_rc[0], pos_rc[1]), 0.5, color='blue')

    particle_positions_rc = xy_to_rc(self.xmin, self.ymin, particle_positions[0,:], particle_positions[1,:], self.res)
    figure, ax = plt.subplots(1)

    # Plot origin at the lower-left corner;
    ax.imshow(map_prob, cmap="gray", origin = "upper")
    # ax.add_patch(circ)
    ax.scatter(particle_positions_rc[1], particle_positions_rc[0], color='red', marker='o', s=1)
    plt.title(title)

    if save_fig:
      plt.savefig(img_name)
      plt.clf()
    else:
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
    if grids.size > 0:
      self.map[grids[0], grids[1]] = np.maximum(self.log_odds_min * np.ones(grids.shape[1]),
                                                self.map[grids[0], grids[1]] + math.log(1 / self.error_ratio))

  def update_occupied(self, grids):
    """
    Update the log-likelihood of the map;
    :param grids: The grids that are observed to be occupied
    :return: None
    """
    if grids.size > 0:
      self.map[grids[0], grids[1]] = np.minimum(self.log_odds_max * np.ones(grids.shape[1]),
                                                self.map[grids[0], grids[1]] + math.log(self.error_ratio))
      # print("The center of log maps is: %s" % (self.map[60, 60]))

  def check_range(self, x, y):
    return x < self.xmax and x > self.xmin and y < self.ymax and y > self.ymin


  def update_log_odds(self, current_lidar_world, robot_pos):
    """
    Update the log_odds of the map
    :param current_lidar_world: The lidar readings in the most recent frame.
    :param robot_pos: The position of the robot;
    :return:
    """

    # print("Before updating log odds the number of occupied vs. free cells are: %s" % (np.unique(self.map, return_counts= True),))
    # Plot obstacles for the current frame;
    num_occupied = 0

    for i in range(current_lidar_world.shape[0]):

      x = current_lidar_world[i, 0]
      y = current_lidar_world[i, 1]

      if self.check_range(x, y):
        # Update free cells;
        num_occupied += 1
        start_rc = xy_to_rc(self.xmin, self.ymin, np.array([robot_pos[0]]), np.array([robot_pos[1]]), self.res)
        end_rc = xy_to_rc(self.xmin, self.ymin, np.array([x]), np.array([y]), self.res)
        grids_free_rc = bresenham2D(start_rc[0, 0], start_rc[1, 0], end_rc[0, 0], end_rc[1, 0])

        # Update free cells; Drop the last one as the last one correspond to the occupied cell.
        self.update_free(grids_free_rc[:, :-1])

        # Update occupied cells;
        self.update_occupied(end_rc)

    # print("The number of occupied cells are: %d" % num_occupied)
    # print("After updating log odds the number of occupied vs. free cells are: %s" % (self.map[np.nonzero(self.map > 0)].size,))
    # print("Done printing values. ")
      # plotting every 12.5 degrees;
      # if i % 50 == 0:
      #   title = "Displaying map after updating lidar ranges from %s to %s." % (LIDAR_ANGLES[0], LIDAR_ANGLES[i])
      #   self.plot(robot_pos, title=title)

