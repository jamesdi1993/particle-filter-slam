from utils.robot_utils import distance_per_tic, yaw_deviation_res
from utils.map_utils import mapCorrelation, tranform_from_body_to_world_frame
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
    self.num_particles = numParticles
    self.n_threshold = particle_threshold

    initial_particle = np.expand_dims(np.append(initial_pos, 1.0/self.num_particles).T, axis = 1) # 4 x 1 array
    self.particles = np.tile(initial_particle, (1, self.num_particles)) # Particles is a 4 x n array [x, y, theta, weight]^T

  def predict_particles(self, encoder_counts, yaw_average, time_elapsed, d_sigma = 0):
    """
    Update the position of the particle from encoder counts and yaw_average;
    :param encoder_counts: The counts of the encoder
    :param yaw_average: The average of yaw velocity
    :param time_elapsed: The time period over which the encoder counts and yaw average is collected
    :param v_sigma: Noise add to the particles, in meter. Gaussian with mean d and sigma d_sigma
    :return: N/A
    """
    d_w = time_elapsed * yaw_average
    d_r = (encoder_counts[0] + encoder_counts[2]) * distance_per_tic / 2
    d_l = (encoder_counts[1] + encoder_counts[3]) * distance_per_tic / 2
    d = (d_r + d_l) / 2  # d = (d_f + d_y) * distance_per_tic / 2

    c = np.cos(self.particles[2, :]) # 1 x n array
    s = np.sin(self.particles[2, :])  # 1 x n array

    # Add noise to the linear displacement;
    d_w_noise = np.random.normal(d, d * d_sigma, self.num_particles)
    dx = d_w_noise * c
    dy = d_w_noise * s

    self.particles[0, :] = self.particles[0, :] + dx
    self.particles[1, :] = self.particles[1, :] + dy
    self.particles[2, :] = (self.particles[2, :] + d_w) % (2 * math.pi)

  def update_particles(self, lidar_body, map, deviation, yaw_deviation):
    """
    Update the weight of the particles based on lidar readings.
    :param lidar_body: The most recent lidar reading in the body frame. 1081 x 2;
    :param map: the current map of the environment
    :param deviation: the size of the area to evaluate correlation for.
    :param yaw_deviation:
    :return: N/A
    """
    x_im = np.arange(map.xmin, map.xmax + map.res, map.res)  # x-positions of each pixel of the map
    y_im = np.arange(map.ymin, map.ymax + map.res, map.res)  # y-positions of each pixel of the map
    binary_map = map.get_binary_map()

    # print("The number of occupied cells vs. free cells: %s" % (np.unique(binary_map, return_counts=True), ))
    # print("The indices of the occupied cells are: %s" % (np.argwhere(binary_map > 0, )))

    correlations = np.zeros((self.num_particles))

    # Temporarily storing all particle weights
    particle_weights = np.copy(self.particles[3, :])


    # Compute map correlation for each particle
    for i in range(self.num_particles):
      pos = self.particles[:-1, i]
      for yaw in np.arange(pos[2] - yaw_deviation / 2, pos[2] + yaw_deviation / 2 + yaw_deviation_res, yaw_deviation_res):
        pos_w_noise = np.copy(pos)
        pos_w_noise[2] = yaw % (2 * math.pi)  # add noise to yaw;
        lidar_world = tranform_from_body_to_world_frame(pos_w_noise, lidar_body)
        # Evaluate for the rectangle centered on the robot position, with half-width correlation_size / 2
        x_range = np.arange(pos[0] - deviation / 2, pos[0] + deviation / 2 + map.res, map.res)
        y_range = np.arange(pos[1] - deviation / 2, pos[1] + deviation / 2 + map.res, map.res)
        c = mapCorrelation(binary_map, x_im, y_im, lidar_world.T, x_range, y_range)
        correlations[i] = np.maximum(correlations[i], np.max(c))

    # Update weight for each particle
    # softmax(z_i) = softmax(z_i - max(z))
    correlations = correlations - np.max(correlations)

    # Update particle weights;
    particle_weights = particle_weights * np.exp(correlations)

    # Normalize
    particle_weights = particle_weights / np.sum(particle_weights)

    # Assign new weights
    self.particles[3, :] = particle_weights

    weights_sum = np.dot(particle_weights, particle_weights.T)

    # TODO: Check if the particles are depleted; Otherwise, resample;
    if 1.0 / weights_sum < self.n_threshold:
      self.resample_particles()


  def get_best_particle_pos(self):
    """
    Get the position of the best particle.
    :return: The position of the best particle.
    """
    best_particle_index = np.argmax(self.particles[3, :])
    return self.particles[:-1, best_particle_index]

  def get_weighted_position(self):
    pass

  def get_particles(self):
    return self.particles

  def get_particles_pos(self):
    return self.particles[:-1, :]

  def resample_particles(self):
    """
    Resample particles, based on the weight of the particles
    :return: N/A
    """
    indices = np.array(range(0, self.num_particles))
    particles_indices = np.random.choice(indices, size=self.num_particles, replace=True, p=self.particles[3, :])
    particles_next = np.zeros((4, self.num_particles))
    for i in range(self.num_particles):
      j = particles_indices[i]
      particles_next[:, i] = self.particles[:, j]
      particles_next[3, i] = 1.0 / self.num_particles # reset weights
    self.particles = particles_next


