from __future__ import division
import cv2
import numpy as np
from gausskern import getNeighborhoodDiffs,calcTempStdDevGetKernel

def astaFilter(frame_window, targetnums):
  """Takes as argument a frame_window which has the current video frame and its surrounding
  frames.  The targetnums argument is a 2d array containing the target number of pixels to
  combine for each pixel in the frame. The function first runs the temporal filter to average 
  the values of each pixel across time.  Then, for each pixel, it will run the spatial
  filter for that pixel at a strength inversely proportional to how many pixels
  could be combined with the temporal filter.  Finally, it returns a 2d array of all the pixels
  for a given video frame calculated by this filter"""

  frame = frame_window.getMainFrame()
  lum = frame[:,:,0]

  (numerators, normalizers), short_of_target = temporalFilter(frame_window,
                                                            targetnums,92)

  #I AM LOSING INFO HERE BY ROUNDING BEFORE THE BILATERAL...PROBLEM?
  temp_filtered_lum = np.rint(numerators / normalizers)

  #put back together bc spatial filter on lum and chrom, not just lum
  frame[:,:,0] = temp_filtered_lum
  
  result_frame = spatialFilter(frame,short_of_target)

  return result_frame


  
def spatialFilter(temp_filtered_frame, distances_short_of_targets):
  """This function chooses a final pixel value with either no
  spatial filtering, or varying degrees of spatial filtering depending
  on how short the temporal filter came to gathering enough pixels"""

 
  #TODO: add a median filtering step before bilateral filter step

  some_filtering = cv2.bilateralFilter(temp_filtered_frame,5,30,0)
  lots_filtering = cv2.bilateralFilter(temp_filtered_frame,7,55,0)

  #need three channels of distances because spatial filter done on all 3  
  dists_short = np.repeat(distances_short_of_targets[:,:,np.newaxis],3,axis=2)


  #this is used as a cutoff for spots where no further filtering required
  min_values = np.zeros_like(dists_short)
  min_values.fill(.1)

  not_short_elems = np.less(dists_short,min_values)

  temp_filter_vals_added = np.where(not_short_elems,temp_filtered_frame,
                                     np.zeros_like(temp_filtered_frame))

  middles = np.zeros_like(dists_short)
  middles.fill(0.45)

  #will be anded with one other numpy array to get middle range
  greater_than_zeros = np.greater_equal(dists_short,min_values)
  less_than_highs = np.less(dists_short,middles)
  a_little_short_elems = np.logical_and(greater_than_zeros,less_than_highs)

  some_space_filter_vals_added = np.where(a_little_short_elems,
                                        some_filtering,temp_filter_vals_added)


  a_lot_short_elems = np.greater_equal(dists_short,middles)
  lots_space_filter_vals_added = np.where(a_lot_short_elems, lots_filtering,
                                         some_space_filter_vals_added)

  return lots_space_filter_vals_added
 
def temporalFilter(frame_window,targetnums, max_error):

#TODO 1:  is there a way to automatically determine max_error rather
#than just I have to figure it out?  may not generalize from one video to a different video
    frame = frame_window.getMainFrame()
    lum = frame[:,:,0]


    kernel_dict = makeGaussianKernels(frame_window)

    filter_keys = nearestFilterKeys(targetnums)

    numerators = 0.0
    normalizers = 0.0

    for i in xrange(0,len(frame_window.frame_list)):

      other_frame = frame_window.frame_list[i]
      other_lum = other_frame[:,:,0]
      
      curr_gauss_weights = getWeightsList(i,kernel_dict)
      frame_distance_weights = np.copy(filter_keys) #need filter_keys later so copy
      makeWeightsArray(frame_distance_weights,curr_gauss_weights) #in-place change
 
      pixel_distance_weights = getNeighborhoodDiffs(lum,other_lum, 50, max_error)
      total_gaussian_weights = pixel_distance_weights * frame_distance_weights

      normalizers += total_gaussian_weights
      numerators += other_lum * total_gaussian_weights

    targets_for_pixels = lookupTargets(filter_keys,kernel_dict)
    distances_short_of_target = targets_for_pixels - normalizers
    
    return (numerators,normalizers), distances_short_of_target


def lookupTargets(filter_keys,kernel_dict):
  lookupTargetVectorized = np.vectorize(lookupOneTarget)
  return lookupTargetVectorized(filter_keys,kernel_dict)


def lookupOneTarget(filter_key,kernel_dict):
  return kernel_dict[filter_key][0]


def makeGaussianKernels(frame_window):
    

    for i in xrange(2,19):  #builds 1-d gaussian kernels of length equal to frame window size 
      kernel_keys.append(i/2)  #with std. devs between .5 and 9.5
      all_kernels.append(calcTempStdDevGetKernel(i/2,frame_window.getLength()))
      

    if frame_window.isFrameAtEdges() != 0: #if near begin or end of video
      all_kernels = rearrangeGaussianKernels(all_kernels,
                                         frame_window.isFrameAtEdges())
   
    kernel_dict = dict(zip(kernel_keys,all_kernels))

    return kernel_dict 


def rearrangeGaussianKernels(all_kernels, distance_off_center):


  """This function is called when the window of surrounding frames
  needed to process a frame is too large to symmetrically take the same
  number of frames from before and after the current frame.  This function
  will rearrange the gaussian kernel in these situations so that the weight
  of each frame still decreases with temporal distance from the current frame."""
 
  resorted_kernels = []
  
  if distance_off_center == 0:
    return all_kernels

  center_index = all_kernels[0][1].size // 2
  index = distance_off_center + center_index
  i = 0
  
  for kernel in all_kernels:

    kernel_list = kernel[1].tolist()
   
    if distance_off_center < 0: #frame is near beginning of video

      dont_sort_edge_index = center_index - index
      dont_sort_part_copy = list(kernel_list[dont_sort_edge_index:center_index])
      del kernel_list[dont_sort_edge_index:center_index]
      kernel_list.sort()
      kernel_list.reverse()
      kernel_list = dont_sort_part_copy + kernel_list

    elif distance_off_center > 0: #frame i/s near end of video
         

      dont_sort_begin_index =  center_index+1
      dont_sort_part_copy = list(kernel_list[dont_sort_begin_index :  
                                      len(kernel_list) - distance_off_center])
      del kernel_list[dont_sort_begin_index : 
                                       len(kernel_list) - distance_off_center]
      kernel_list.sort() #no reverse in this case
      kernel_list = kernel_list + dont_sort_part_copy
   
    resorted_kernels.append((kernel[0],np.array(kernel_list)))

  return resorted_kernels


def getWeightsList(index,kernel_dict):
  """This function will return the gaussian distance weights based on index,
  which is both the frame number in the queue and the index to the proper
  gaussian weight"""
  weights_list = []
  sorted_keys = sorted(kernel_dict.iterkeys())
  #go through dict in order
  for key in sorted_keys:
    weights_list.append(kernel_dict[key][1].item(index))

  return weights_list


def makeWeightsArray(filter_keys, weights_list):
  """Takes a numpy array of equal dimension to the pixel lum array filled with values of about 
  how many pixels need to be combined at each pixel (filter_keys).  Associates these with
  the correct elements in weights_list which holds the gaussian weights for the different
  filter_keys.  Will return a numpy array of pixel lum size with values of spatial gaussian weights"""

  weights_list.reverse()

  filter_keys[filter_keys > 8.6] = weights_list[0] #9.0 weight
  filter_keys[filter_keys > 8.1] = weights_list[1] #8.5 weight
  filter_keys[filter_keys > 7.6] = weights_list[2] #8.0 weight
  filter_keys[filter_keys > 7.1] = weights_list[3] #7.5 weight
  filter_keys[filter_keys > 6.6] = weights_list[4] #7.0 weight
  filter_keys[filter_keys > 6.1] = weights_list[5] #6.5 weight
  filter_keys[filter_keys > 5.6] = weights_list[6] #6.0 weight
  filter_keys[filter_keys > 5.1] = weights_list[7] #5.5 weight
  filter_keys[filter_keys > 4.6] = weights_list[8] #5.0 weight
  filter_keys[filter_keys > 4.1] = weights_list[9] #4.5 weight
  filter_keys[filter_keys > 3.6] = weights_list[10] #4.0 weight
  filter_keys[filter_keys > 3.1] = weights_list[11] #3.5 weight
  filter_keys[filter_keys > 2.6] = weights_list[12] #3.0 weight
  filter_keys[filter_keys > 2.1] = weights_list[13] #2.5 weight
  filter_keys[filter_keys > 1.6] = weights_list[14] #2.0 weight
  filter_keys[filter_keys > 1.1] = weights_list[15] #1.5 weight
  filter_keys[filter_keys == 1.0] = weights_list[16] #1.0 weight
  
  return filter_keys

def nearestFilterKeys(target_nums):
  """Keys in the filter dict are at 1, 1.5, 2, etc..."""
  return np.round(target_nums * 2) / 2
  
