## Main files

* slam_particle_filter.py: Main file for time-correlation, coordination between mapping, particle update and texture map;
** Example usage: python slam_particle_filter.py 0.1 0.05 100 1 0 23 #sigma_d, sigma_yaw, number_of_particles, save_image, texture_map, dataset_index
* load_data.py: File for loading encoder, imu, lidar and image data;
* map.py: File for manipulating with map, such as updating and visualization;
* robot_pos.py: File for manipulating with the particles;
* Texture mapping.ipynb: Experimentation with texture mapping;
* sigma_process_utils.py: Helper function for signal processing, such as low-pass filter;
* robot_utils.py: Robot hardware configurations;
* map_utils.py: Helper functions for correlation and transformation;
* camera_utils.py: Helper functions for transforming RGB and depth images.