from mpl_toolkits.mplot3d import Axes3D
from utils.robot_utils import P_LIDAR_TO_BODY, LIDAR_MIN_JITTER, LIDAR_MAX
from load_data import load_data

import numpy as np
import math
import matplotlib.pyplot as plt; plt.ion()
import time

def tic():
  return time.time()

def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT 
  im              the map 
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
  xs,ys           physical x,y,positions you want to evaluate "correlation" 

  OUTPUT 
  c               sum of the cell values of all the positions hit by range sensor
  '''
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])

      # print("The offsets are x: %s; y: %s" % (xs[jx], ys[jy]))
      # # indices = np.vstack((iy, ix))
      # # print("The number of laser hit cells are: %s" % (iy.shape,))
      # print("The correlation is: %s" % cpr[jx,jy])
      # print("The valid cells are:  %s" % (np.argwhere(valid.astype(int) > 0), ))
      # print("The number of unique values of valid cells are: %s" % (np.unique(valid.astype(int), return_counts=True),))
      # print("The number of valid cells are: %s" % (ix[valid].shape, ))
  return cpr


def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1, 1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y)).astype(int)
    

def test_bresenham2D():
  import time
  sx = 0
  sy = 1
  print("Testing bresenham2D...")
  r1 = bresenham2D(sx, sy, 10, 5)
  r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
  r2 = bresenham2D(sx, sy, 9, 6)
  r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])	
  if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
    print("...Test passed.")
  else:
    print("...Test failed.")

  # Timing for 1000 random rays
  num_rep = 1000
  start_time = time.time()
  for i in range(0,num_rep):
	  x,y = bresenham2D(sx, sy, 500, 200)
  print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))

def test_mapCorrelation():
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = np.load("test_ranges.npy")

  # take valid indices
  indValid = np.logical_and((ranges < 30),(ranges> 0.1))
  ranges = ranges[indValid]
  angles = angles[indValid]

  # init MAP
  MAP = {}
  MAP['res']   = 0.05 #meters
  MAP['xmin']  = -20  #meters
  MAP['ymin']  = -20
  MAP['xmax']  =  20
  MAP['ymax']  =  20 
  MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
  MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
  MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8

  
  # xy position in the sensor frame
  xs0 = ranges*np.cos(angles)
  ys0 = ranges*np.sin(angles)
  
  # convert position in the map frame here 
  Y = np.stack((xs0,ys0))
  
  # convert from meters to cells
  xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
  yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
  
  # build an arbitrary map 
  indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
  MAP['map'][xis[indGood[0]],yis[indGood[0]]]=1

  #import pdb
  #pdb.set_trace()
      
  x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
  y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

  x_range = np.arange(-0.2,0.2+0.05,0.05)
  y_range = np.arange(-0.2,0.2+0.05,0.05)


  
  print("Testing map_correlation with {}x{} cells".format(MAP['sizex'],MAP['sizey']))
  ts = tic()
  c = mapCorrelation(MAP['map'],x_im,y_im,Y,x_range,y_range)
  toc(ts,"Map Correlation")

  c_ex = np.array([[3,4,8,162,270,132,18,1,0],
		  [25  ,1   ,8   ,201  ,307 ,109 ,5  ,1   ,3],
		  [314 ,198 ,91  ,263  ,366 ,73  ,5  ,6   ,6],
		  [130 ,267 ,360 ,660  ,606 ,87  ,17 ,15  ,9],
		  [17  ,28  ,95  ,618  ,668 ,370 ,271,136 ,30],
		  [9   ,10  ,64  ,404  ,229 ,90  ,205,308 ,323],
		  [5   ,16  ,101 ,360  ,152 ,5   ,1  ,24  ,102],
		  [7   ,30  ,131 ,309  ,105 ,8   ,4  ,4   ,2],
		  [16  ,55  ,138 ,274  ,75  ,11  ,6  ,6   ,3]])
    
  if np.sum(c==c_ex) == np.size(c_ex):
	  print("...Test passed.")
  else:
	  print("...Test failed. Close figures to continue tests.")	

  #plot original lidar points
  fig1 = plt.figure()
  plt.plot(xs0,ys0,'.k')

  #plot map
  fig2 = plt.figure()
  plt.imshow(MAP['map'],cmap="hot");

  #plot correlation
  fig3 = plt.figure()
  ax3 = fig3.gca(projection='3d')
  X, Y = np.meshgrid(np.arange(0,9), np.arange(0,9))
  ax3.plot_surface(X,Y,c,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
  plt.show()
  
  
def show_lidar_with_file(filename):
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = np.load(filename)
  ax = plt.subplot(111, projection='polar')
  ax.plot(angles, ranges)
  ax.set_rmax(10)
  ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
  ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
  ax.grid(True)
  ax.set_title("Lidar scan data", va='bottom')
  plt.show()

def show_lidar_with_ranges(ranges):
  angles = np.arange(-135, 135.25, 0.25) * np.pi/180.0
  ax = plt.subplot(111, projection='polar')
  ax.plot(angles, ranges)
  ax.set_rmax(10)
  ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
  ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
  ax.grid(True)
  ax.set_title("Lidar scan data", va='bottom')
  plt.show()

def transform_to_lidar_frame(distances, angles, lidar_min, lidar_max):
  """
  Transform the lidar distances into the coordinates of lidar frame; xy coordinates
  :param distances: The distances of the lidar scan. a 1081 1-dimension array
  :param angles: The angles for the lidar; a 1081 1-dimension array
  :return: Coordinates of the closest obstacle detected by each laser, a 1081 x 2 array
  """
  valid = np.logical_and(distances > lidar_min, distances < lidar_max)
  distances_valid = distances[valid]
  angles_valid = angles[valid]

  xs = distances_valid * np.cos(angles_valid) # xs = d * cos(\theta)
  ys = distances_valid * np.sin(angles_valid) # ys = d * sin(\theta)

  coords = np.vstack((xs, ys)).T
  # print("The shape of coords is: %s" % (coords.shape,))
  # print("The first 5 coordinates are: %s" % coords[:5, 0, :])
  return coords


def transform_from_lidar_to_body_frame(coordinates):
  """
  Transform coordinates in lidar frame to body frame;
  :param coordinates: The coordinates in lidar frame, a n x 2 array
  :return: The coordinates in robot body frame, a n x 2 array
  """
  coords_hom = to_homogenuous(coordinates)
  l_to_b_matrix = np.identity(coordinates.shape[1] + 1) # 3 x 3 matrix
  l_to_b_matrix[-1, 0] = P_LIDAR_TO_BODY[0]
  l_to_b_matrix[-1, 1] = P_LIDAR_TO_BODY[1]

  # print("print the first five coordinates: %s" % transformed_coords[:5, :])
  return from_homogenuous(coords_hom.dot(l_to_b_matrix))

def tranform_from_body_to_world_frame(pos, coords):
  """
  Transform coordinates from the body to the world frame, specified by pos.
  :param pos: The position of the robot. 1 x 2 array
  :param coords: The coordinates to be transformed; n x 2 array
  :return: The coordinates in the world frame. 2 x n array
  """
  coords_hom = to_homogenuous(coords) # copy n x 3
  r_to_w_matrix = np.identity(pos.shape[0]) # 3 x 3 matrix
  # p
  r_to_w_matrix[0, -1] = pos[0]
  r_to_w_matrix[1, -1] = pos[1]
  # rotation in 2D
  c, s = math.cos(pos[-1]), math.sin(pos[-1])
  r_to_w_matrix[0:2, 0:2] = np.array([[c, -s], [s, c]])
  c_transformed_hom = np.dot(r_to_w_matrix, coords_hom.T) # 3 x n matrix
  return from_homogenuous(c_transformed_hom.T)

def to_homogenuous(coords_euclid):
  """
  Transform a coordinate from euclidean to homogenuous coordinate.
  :param coords_euclid: euclidean coordinates.  A m x d array
  :return: homogenuous coordinate. A m x (d + 1) array
  """
  # print("The shape of the coordinates is: %s" % (coords_euclid.shape, ))
  return np.append(coords_euclid, np.ones((coords_euclid.shape[0], 1)), axis = 1);

def from_homogenuous(coords_hom):
  """
  Transform a homogenuous coordinate into an euclidean coordinate.
  :param coords_hom: the homogenuous coordinate; A n x (d + 1) array
  :return: An euclidean coordinate. A n x d array
  """
  return np.divide(coords_hom[:, :-1], coords_hom[:, -1:])


# convert xy to rc coordinate;
# def xy_to_rc(x_range, y_range, x, y, res):
#   rows = (np.absolute(y - y_range/2)/res).astype(int)
#   cols = ((x_range/2 + x)/res).astype(int)
#   return np.vstack((rows, cols))

def xy_to_rc(x_min, y_min, x, y, res):
  rows = np.int16(np.round((x - x_min) / res))
  cols = np.int16(np.round((y - y_min) / res))
  return np.vstack((rows, cols))

def recover_from_log_odds(x):
  return 1 - (1 / (1 + np.exp(x)))

def test_body_to_world():
  print("Testing for transform_from_body_to_world frame")
  r_pos = np.array([1, 1, math.pi / 6])
  coordinates = np.array([[2, 0], [3.0/2, -math.sqrt(3)/2], [0, 0]])
  e_coordinates = np.array([[1 + math.sqrt(3), 2], [1 + math.sqrt(3), 1], [1, 1]]).T
  transformed_coordinates = tranform_from_body_to_world_frame(r_pos, coordinates)
  if np.sum(transformed_coordinates == e_coordinates) == np.size(e_coordinates):
    print("...Test passed.")
  else:
    print("...Test failed.")
    print("Transformed coordinates are: %s" % transformed_coordinates)
    print("Expected coordinates are: %s" % e_coordinates)

def test_lidar_to_body():
  print("Testing for transforming from_lidar_to_body_frame():")
  coordinates = np.array([[1, 0], [0, 1], [1, 1]])
  e_coordinates = np.array([[1.29833, 0], [0.29833, 1], [1.29833, 1]])
  transformed_coordinates = transform_from_lidar_to_body_frame(coordinates)

  if np.sum(transformed_coordinates == e_coordinates) == np.size(e_coordinates):
    print("...Test passed.")
  else:
    print("...Test failed.")
    print("Transformed coordinates are: %s" % transformed_coordinates)
    print("Expected coordinates are: %s" % e_coordinates)

if __name__ == '__main__':
  # dataset_index = 20
  # filename = "Hokuyo%d.npz" % dataset_index
  # filename = "test_ranges.npy"
  data = load_data(dataset_index=20)
  ranges = data['lidar_ranges'][:, 0]

  show_lidar_with_ranges(ranges)
  test_mapCorrelation()
  test_bresenham2D()
  test_lidar_to_body()
  test_body_to_world()
  print("Finished testing. ")
  
