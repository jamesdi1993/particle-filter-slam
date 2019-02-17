from utils.robot_utils import distance_per_tic
import math
import numpy as np

class Particle():

  def __init__(self, pos, weight):
    self.pos = np.array(pos).astype(float) #x, y, \theta
    self.weight = weight

  def update_weight(self, weight):
    self.weight = weight

  def update_pos(self, pos):
    self.pos = pos

  def update_pos_with_measurements(self, encoder_counts, yaw_average, time_elapsed):
    """
    Update the position of the particle from encoder counts and yaw_average;
    :param encoder_counts: The counts of the encoder
    :param yaw_average: The average of yaw velocity
    :param time_elapsed: The time period over which the encoder counts and yaw average is collected
    :return: N/A
    """
    d_w = time_elapsed * yaw_average
    d_r = (encoder_counts[0] + encoder_counts[2]) * distance_per_tic / 2
    d_l = (encoder_counts[1] + encoder_counts[3]) * distance_per_tic / 2
    self.update_pos_with_distances(d_l, d_r, d_w)

  def update_pos_with_distances(self, left_d, right_d, angular_d):
    """
    Update the position of the particle.
    :param left_d: The displacement of the left wheel
    :param right_d: The displacement of the right wheel
    :param angular_d: The delta of the angle;
    :return: N/A
    """
    d = (left_d + right_d) / 2 # d = (d_f + d_y) * distance_per_tic / 2
    dx, dy = d * math.cos(self.pos[2]), d * math.sin(self.pos[2])
    self.pos[0] = self.pos[0] + dx  # x = d * cos(\theta);
    self.pos[1] = self.pos[1] + dy  # y = d * sin(\theta)
    self.pos[2] = (self.pos[2] + angular_d) % (2 * math.pi)

  def get_pos(self):
    return self.pos

  def __repr__(self):
    return "Position: %s; \n Weight: %s" % (self.pos.T, self.weight)

  def __str__(self):
    return "Position: %s; \n Weight: %s" % (self.pos.T, self.weight)

class RobotPos():

  def __init__(self, initial_pos, numParticles = 1):
    # self.position = np.array(initial_pos)  # estimated position of the robot, a 3 x 1 array
    self.particles = [Particle(initial_pos, float(1)/numParticles) for x in range(numParticles)] # initialize particle
    self.num_particles = numParticles
    print("The particles are: %s" % (self.particles[0]))

  def predict_particles(self, encoder_counts, yaw_average, time_elapsed):
    """
    Update the position of the particle from encoder counts and yaw_average;
    :param encoder_counts: The counts of the encoder
    :param yaw_average: The average of yaw velocity
    :param time_elapsed: The time period over which the encoder counts and yaw average is collected
    :return: N/A
    """
    for particle in self.particles:
      particle.update_pos_with_measurements(encoder_counts, yaw_average, time_elapsed)

  def get_best_particle(self):
    """
    Get the particle of the highest weight.
    :return: The particle of the highest weight.
    """
    highest_weight = 0
    best_particle = None
    for particle in self.particles:
      if particle.weight > highest_weight:
        best_particle = particle
        highest_weight = particle.weight
    return best_particle

  def get_best_particle_pos(self):
    """
    Get the position of the best particle.
    :return: The position of the best particle.
    """
    best_particle = self.get_best_particle()
    return best_particle.get_pos()

  def get_weighted_position(self):
    """
    Get the weighted average of the particle set
    :return: The best guess of the robot position.
    """
    pos = np.array([0,0,0])
    weights = 0
    for particle in self.particles:
      pos += particle.weight * particle.get_pos()
      weights += particle.weight

    # Normalized weight;
    return pos / weights
