import numpy.linalg as linalg
import numpy as np
import math

def to_homogenuous(coords_euclid):
  return np.append(coords_euclid, np.ones((1,1)), axis = 0);

def from_homog(coordinate):
	return coordinate[:-1] / coordinate[-1]

roll = 0
pitch = 0.36
yaw = 0.021

f_su=585.05108211
f_sv=585.05108211
f_s_theta =0
cu=242.94140713
cv=315.83800193


K = np.array([[f_su, f_s_theta, cu], [0, f_sv, cv], [0, 0, 1]])
projection_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

K_inv = linalg.inv(K)
# projection_matrix_inv = linalg.inv(projection_matrix)

R_c_to_o = np.array([[0, -1, 0],
                     [0, 0, -1],
                     [1, 0, 0]])

R_x = np.array([[1,0,0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]])

R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]])

R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]])

R_c_to_b = R_z.dot(R_y.dot(R_x))


p_c_to_b = np.array([[0.18], [0.005], [0.36]])

pose_c_b = np.identity(4)
pose_c_b[0:3, 0:3] = R_c_to_o.dot(R_c_to_b.T)
pose_c_b[0:3, -1] = -R_c_to_o.dot(R_c_to_b.T).dot(p_c_to_b).reshape(3)

p_c = to_homogenuous(p_c_to_b)

pose_c_b_inv = linalg.inv(pose_c_b)

# def tranform_from_2D_to_3D(coordinate, depth):
#     """
#     Coordinate is a 3 x 1 in homogeneuous coordinate
#     """
#     c_euclid = from_homog(coordinate)
#     c_3D = np.ones((3,1)) # 3 x 1 array
#     c_3D[0:2, :] = c_euclid
#     c_3D = depth * c_3D
#     return to_homogenuous(c_3D)

def tranform_from_2D_to_3D(coordinate, depth):
    """
    Coordinate is a 3 x 1 in homogeneuous coordinate
    """
    # TODO: Revisit this logic;
    return to_homogenuous(coordinate * depth)



def transform_from_camera_to_body_frame(coordinate):
    """
    Coordinate is a 2 x 1 in euclidean coordinate
    """
    c_projected = K_inv.dot(to_homogenuous(coordinate[0:2]))
    c_o = to_homogenuous(c_projected * coordinate[-1])
    return pose_c_b_inv.dot(c_o)