from utils.robot_utils import distance_per_tic
import math
import numpy as np

class Particle():

  def __init__(self):
    self.pos = np.array([[0],[0],[0]]) #x, y, \theta
    self.weight = 1

  def update_weight(self, weight):
    self.weight = weight

  def update_pos(self, pos):
    self.pos = pos

  def update_pos(self, left_d, right_d, angular_d):
    """
    Update the position of the particle.
    :param left_d: The displacement of the left wheel
    :param right_d: The displacement of the right wheel
    :param angular_d: The delta of the angle;
    :return:
    """
    d = (left_d + right_d) * distance_per_tic / 2 # d = (d_f + d_y) * distance_per_tic / 2
    dx, dy = d * math.cos(self.pos[2]), d * math.sin(self.pos[2])
    self.pos[0] = self.pos[0] + dx  # x = d * cos(\theta);
    self.pos[1] = self.pos[1] + dy  # y = d * sin(\theta)
    self.pos[2] = self.pos[2] + angular_d


  def __repr__(self):
    return "Position: %s; \n Weight: %s" % (self.pos.T, self.weight)

  def __str__(self):
    return "Position: %s; \n Weight: %s" % (self.pos.T, self.weight)

class RobotPos():

  def __init__(self, numParticles = 1):
    self.position = np.array([[0],[0],[0]])  # estimated position of the robot, a 3 x 1 array
    self.particles = [Particle() for x in range(numParticles)] # initialize particle
    print("The particles are: %s" % (self.particles[0]))

  def predict_particles(self, left_d, right_d, angular_d):
    """
    Update the positions of the each particle
    :param left_d:
    :param right_d:
    :param angular_d:
    :return:
    """
    self.particles = [particle.update_pos(left_d, right_d, angular_d) for particle in self.particles]

