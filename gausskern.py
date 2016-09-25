from __future__ import division
import cv2
import numpy as np
import sys
#if I normalize here like I did in neighborhood kernel I think I would
#just be doing it twice but it wouldnt affect correctness because im normalizing
#again anyways
def get1DKernel(size, std_dev):
  
  if size % 2 == 0:
    size += 1
  kernel_t = cv2.getGaussianKernel(size,std_dev)
  return kernel_t


def getKernelCenter(kernel):
  return kernel.item(len(kernel) // 2)


def calcTempStdDevGetKernel(target_num,window_size):
  """This function will make it so if all temporal pixels had identical
  neighborhoods, the contribution of the neighborhood pixels would be 
  equal to 2 * target_num * G_center where G_center is the weight on center
  pixel.  Returns a pair of the kernel as well as the scaled target"""
  #I have attenuation at 34 so I need to handle target_nums of up to
  #to 10.  If I had a much greater attenuation, I would need to change this
  #algorithm
  
  if target_num > 9:
    sys.stderr.write("Mapping should not go over 9")
    sys.exit()

  #paper changes both neighborhood size and std dev dynamically..
  #I have a fixed size neighborhood and just change std dev
  if window_size < 19: #if I want smaller window must change atten
    sys.stderr.write("window size is too small to handle all cases")
    sys.exit()

  temp_std_dev = 0.5
  kernel = get1DKernel(window_size, temp_std_dev)
  target_weighted = 2 * target_num * getKernelCenter(kernel) * 1.0    #1.0 is because center has
                                                                      #perfect match with itself 
  neighborhood_weight = kernel.sum() - getKernelCenter(kernel)

  while abs(neighborhood_weight - target_weighted) > .05:

    temp_std_dev += 0.01
    kernel = get1DKernel(window_size, temp_std_dev)
    target_weighted = 2 * target_num * getKernelCenter(kernel) * 1.0
    neighborhood_weight = kernel.sum() - getKernelCenter(kernel)
  
  return target_weighted, kernel


def get2DKernel(size, std_dev):
  """Returns a kernel of size by size with standard deviation given in other arg"""
  if size % 2 == 0:
    size += 1
  #adapted from Howse book
  kernel_x = cv2.getGaussianKernel(size, std_dev)
  kernel_y = cv2.getGaussianKernel(size, std_dev)
  kernel = kernel_y * kernel_x.T
  return kernel

#TODO: I am normalizing before the operation in getting neighborhood
 #, is this wrong?
def getNeighborhoodCompareKernel(size, std_dev):
  """Calls get2DKernel to create a gaussian neighborhood comparison kernel.
  It sets center pixel to zero because a pixel is not included in its neighborhood 
  (think shot noise)"""
  kernel = get2DKernel(size, std_dev)
  middle_idx = size / 2
  # just compare neighborhoods, leave center pixel out
  kernel[middle_idx][middle_idx] = 0.0
  kernel = kernel / kernel.sum() #normalize
  return kernel


def getNeighborhoodDiffs(neighborhood_1, neighborhood_2,min_diff,max_diff):
  """This function will calculate the diffs between two
  numpy array images (lums) passed as arguments at every pixel. Then it will scale
  the result to be between zero and one. Assume that below some threshold
  (min_diff) the neighborhoods should be considered as identical and above
  some threshold (max_diff) the neighborhoods will be considered as different
  as they possibly can be (returns 0)"""

  neighborhood_diffs = np.abs(neighborhood_1 - neighborhood_2)

  #from paper: "The neighborhood size, often between 3 and 5, can be 
  #varied depending on noise as can [std_dev] (usually between 2 and 6
  g_kernel = getNeighborhoodCompareKernel(5,2) 

  #TODO: do i really want BORDER_REPLICATE?
  neigh_diffs = np.array(neighborhood_diffs,dtype='float64')
  diffs_each_pixel = cv2.filter2D(neigh_diffs,
                       -1,g_kernel,borderType=cv2.BORDER_REPLICATE)
  min_diffs = np.empty_like(diffs_each_pixel)
  max_diffs = np.empty_like(diffs_each_pixel)
  
  min_diffs.fill(min_diff)
  max_diffs.fill(max_diff)

  values = np.zeros_like(diffs_each_pixel)
  values = distanceMetric(diffs_each_pixel,min_diffs,max_diffs)
  return values

def distanceMetric(distance,min,max):

  distance = distance - min
  max = max - min
  zeros = np.zeros_like(distance)

  a = np.less(distance,zeros)
  dist = np.where(a,zeros,distance)

  b = np.greater(dist,max)
  dis = np.where(b,max,dist)

  return (max - dis) / max


def distanceMetric2(distance,min,max):
  """Sequential distance metric.  Not used in final program
  for performance reasons"""
  distance = distance - min
  max = max - min

  if (distance < 0):
    distance = 0
  if (distance > max):
    distance = max

  return (max - distance) / max

if __name__ == "__main__":
  for i in xrange(1, 20):
    print "MIN 4, MAX 16, ARG ",str(i),"=",distanceMetric(i,4,16)
    
    print "MIN 4, MAX 16, ARG ",str(i),".5=",distanceMetric(i+.5,4,16)

  std_dev = 2 
#  kernel_t = cv2.getGaussianKernel(9,std_dev)

  
#  a = getNeighborhoodCompareKernel(4, 2)
  
  j = np.array(([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]))
  k = np.array(([[3,5,2],[9,5,6],[1,8,9]]))
  print j
  l = np.array(([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]))

  print k
  #should be zero in each (which is one) 
  print  getNeighborhoodDiffs(j,l,4, 16)


  #getNeighborhoodDiffs(l,j)
  #these approximate how for off I will be and still want to take it
  #almost completely
  m = np.array(([[1,2,3,4,5],[4,8,8,9,10],[11,12,13,14,15]]))

  n = np.array(([[1,2,3,3,4],[6,7,8,9,10],[11,12,13,14,15]]))

  print m,n
#  print getNeighborhoodDiffs(m,n,2, 8)

#  getNeighborhoodDiffs(n,m,2,8)

  #p and q are so far apart that should be almost zero   
  p = np.array(([[1,2,3,4,5],[4,8,8,9,10],[11,12,13,14,15]]))

  q = p + 20 

  print p
  print q
  
#  print getNeighborhoodDiffs(p,q,2,8)
#  print getNeighborhoodDiffs(q,p,2,8)

  s =  np.array(([[1,2,11,10,5],[12,7,8,9,10],[11,12,13,14,15]]))
 
  r =  np.array(([[15,14,11,11,11],[6,7,8,8,6],[5,4,12,12,1]]))
  print s
  print r
#  print  getNeighborhoodDiffs(s,r,4,9)
#  print   getNeighborhoodDiffs(r,s,4,9)
  print "EEE"
  print calcTempStdDevGetKernel(2.0,19)
