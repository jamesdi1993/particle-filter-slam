from utils.robot_utils import distance_per_tic
from utils.map_utils import mapCorrelation
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

  def update_pos_with_measurements(self, encoder_counts, yaw_average, time_elapsed, d_sigma):
    """
    Update the position of the particle from encoder counts and yaw_average;
    :param encoder_counts: The counts of the encoder
    :param yaw_average: The average of yaw velocity
    :param time_elapsed: The time period over which the encoder counts and yaw average is collected
    :param d_sigma: The sigma added to the distance; per 0.025 sec.
    :return: N/A
    """
    d_w = time_elapsed * yaw_average
    d_r = (encoder_counts[0] + encoder_counts[2]) * distance_per_tic / 2
    d_l = (encoder_counts[1] + encoder_counts[3]) * distance_per_tic / 2
    self.update_pos_with_distances(d_l, d_r, d_w, d_sigma)

  def update_pos_with_distances(self, left_d, right_d, angular_d, d_sigma):
    """
    Update the position of the particle.
    :param left_d: The displacement of the left wheel
    :param right_d: The displacement of the right wheel
    :param angular_d: The delta of the angle;
    :param d_sigma: The sigma added to the distance; per 0.025 sec.
    :return: N/A
    """
    d = (left_d + right_d) / 2 # d = (d_f + d_y) * distance_per_tic / 2
    d_w_noise = np.random.normal(d, d * d_sigma)
    dx, dy = d_w_noise * math.cos(self.pos[2]), d_w_noise * math.sin(self.pos[2])
    self.pos[0] = self.pos[0] + dx  # x = d * cos(\theta);
    self.pos[1] = self.pos[1] + dy  # y = d * sin(\theta)
    self.pos[2] = (self.pos[2] + angular_d) % (2 * math.pi)

  def get_pos(self):
    return self.pos

  def get_weight(self):
    return self.weight

  def __repr__(self):
    return "Position: %s; \n Weight: %s" % (self.pos.T, self.weight)

  def __str__(self):
    return "Position: %s; \n Weight: %s" % (self.pos.T, self.weight)

class RobotPos():

  def __init__(self, initial_pos, numParticles = 1, particle_threshold = 1):
    # self.position = np.array(initial_pos)  # estimated position of the robot, a 3 x 1 array
    self.particles = [Particle(initial_pos, float(1)/numParticles) for x in range(numParticles)] # initialize particle
    self.num_particles = numParticles
    self.n_threshold = particle_threshold
    print("The particles are: %s" % (self.particles[0]))

  def predict_particles(self, encoder_counts, yaw_average, time_elapsed, d_sigma = 0):
    """
    Update the position of the particle from encoder counts and yaw_average;
    :param encoder_counts: The counts of the encoder
    :param yaw_average: The average of yaw velocity
    :param time_elapsed: The time period over which the encoder counts and yaw average is collected
    :param v_sigma: Noise add to the particles, in meter. Gaussian with mean d and sigma d_sigma
    :return: N/A
    """
    for particle in self.particles:
      particle.update_pos_with_measurements(encoder_counts, yaw_average, time_elapsed, d_sigma)

  def update_particles(self, lidar_reading, map, deviation):
    """
    Update the weight of the particles based on lidar readings.
    :param lidar_reading: The most recent lidar reading in the world frame. 1081 x 2;
    :param map: the current map of the environment
    :param deviation: the size of the area to evaluate correlation for.
    :return: N/A
    """
    x_im = np.arange(map.xmin, map.xmax + map.res, map.res)  # x-positions of each pixel of the map
    y_im = np.arange(map.ymin, map.ymax + map.res, map.res)  # y-positions of each pixel of the map
    binary_map = map.get_binary_map()

    # print("The number of occupied cells vs. free cells: %s" % (np.unique(binary_map, return_counts=True), ))
    # print("The indices of the occupied cells are: %s" % (np.argwhere(binary_map > 0, )))

    correlations = np.zeros((len(self.particles)))

    # Temporarily storing all particle weights
    particle_weights = np.zeros(len(self.particles))

    # Compute map correlation for each particle
    for i in range(len(self.particles)):
      particle = self.particles[i]
      pos = particle.get_pos()  # robot_pos, in the world frame
      # Evaluate for the rectangle centered on the robot position, with half-width correlation_size / 2
      x_range = np.arange(pos[0] - deviation / 2, pos[0] + deviation / 2 + map.res, map.res)
      y_range = np.arange(pos[1] - deviation / 2, pos[1] + deviation / 2 + map.res, map.res)
      c = mapCorrelation(binary_map, x_im, y_im, lidar_reading.T, x_range, y_range)
      correlations[i] = np.max(c)
      particle_weights[i] = particle.get_weight()

    # Update weight for each particle
    # softmax(z_i) = softmax(z_i - max(z))
    correlations = correlations - np.max(correlations)

    # Update particle weights;
    particle_weights = particle_weights * np.exp(correlations)

    # Normalize
    particle_weights = particle_weights / np.sum(particle_weights)

    # Assign new weights
    for j in range(len(self.particles)):
      particle = self.particles[j]
      particle.update_weight(particle_weights[j])

    weights_sum = np.dot(particle_weights, particle_weights.T)

    # TODO: Check if the particles are depleted; Otherwise, resample;
    if 1.0 / weights_sum < self.n_threshold:
      self.resample_particles()

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

  def get_particles(self):
    return self.particles

  def get_particles_pos(self):
    particle_positions = np.zeros((3, len(self.particles))).astype(float)
    for i in range(len(self.particles)):
      particle_positions[:, i] = self.particles[i].get_pos()
    return particle_positions

  def resample_particles(self):
    """
    Resample particles, based on the weight of the particles
    :return: N/A
    """
    particle_weights = np.array([particle.get_weight() for particle in self.particles])
    indices = np.array(range(0, len(self.particles)))
    particles_indices = np.random.choice(indices, size=self.num_particles, replace=True, p=particle_weights)
    particles_next = [Particle(self.particles[i].get_pos(), 1.0 / self.num_particles) for i in particles_indices]
    self.particles = particles_next


