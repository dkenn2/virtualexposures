from __future__ import division
import cv2
import sys
import faulthandler
import numpy as np
from tonemap import findTargetLuminances,tonemapSpatiallyUniform
from gausskern import getNeighborhoodDiffs,calcTempStdDevGetKernel

def astaFilter(frame_window, targetnums):
  frame = frame_window.getMainFrame()
  lum = frame[:,:,0]

  (numerators, normalizers), short_of_target = temporalFilter(frame_window,
                                                            targetnums,92)
  print "ORIGINAL: ",lum

  #I AM LOSING INFO HERE BY ROUNDING BEFORE THE BILATERAL...PROBLEM?
  temp_filtered_lum = np.rint(numerators / normalizers)

  #put back together bc spatial filter on lum and chrom, not just lum
  frame[:,:,0] = temp_filtered_lum
  
  print "TEMP_FILTERED: ",frame[:,:,0] 

  result_frame = spatialFilter(frame,short_of_target)

  print "TEMP_AND_SPACE_FILTERED: ", result_frame[:,:,0]

  return result_frame
  
def spatialFilter(temp_filtered_frame, distances_short_of_targets):
  """This function chooses a value for the final value with either no
  spatial filtering, or varying degrees of spatial filtering depending
  on how short the temporal filter came to gathering enough pixels"""

 
  #TODO: add a median filtering step before bilateral filter step
  some_filtering = cv2.bilateralFilter(temp_filtered_frame,5,30,0)
  lots_filtering = cv2.bilateralFilter(temp_filtered_frame,7,55,0)

  #need three channels of distances short because spatial filter 
  #is on all channels
   
  dists_short = np.repeat(distances_short_of_targets[:,:,np.newaxis],3,axis=2)
  #this is used as a cutoff for spots where no further filtering required
  min_values = np.zeros_like(dists_short)
  min_values.fill(.1)
  not_short_elems = np.less(dists_short,min_values)
  temp_filter_vals_added = np.where(not_short_elems,temp_filtered_frame,
                                     np.zeros_like(temp_filtered_frame))

  print "FIRST: ", np.count_nonzero(temp_filter_vals_added)

  middles = np.zeros_like(dists_short)
  middles.fill(0.45)

  #will be anded with one other numpy array to get middle range
  greater_than_zeros = np.greater_equal(dists_short,min_values)
  less_than_highs = np.less(dists_short,middles)
  a_little_short_elems = np.logical_and(greater_than_zeros,less_than_highs)

  some_space_filter_vals_added = np.where(a_little_short_elems,
                                        some_filtering,temp_filter_vals_added)


  
  print "SECOND: ", np.count_nonzero(some_space_filter_vals_added)
  

  a_lot_short_elems = np.greater_equal(dists_short,middles)
  lots_space_filter_vals_added = np.where(a_lot_short_elems, lots_filtering,
                                         some_space_filter_vals_added)

  print "THIRD: ", np.count_nonzero(lots_space_filter_vals_added)

  return lots_space_filter_vals_added
 
def temporalFilter(frame_window,targetnums, max_error):

#TODO 1:  is there a way to automatically determine max_error rather
#than just I have to figure it out?  may not generalize from one video to a different video
    frame = frame_window.getMainFrame()
    lum = frame[:,:,0]

    kernel_keys = []
    all_kernels = []

    for i in xrange(2,19):
      kernel_keys.append(i/2)
      all_kernels.append(calcTempStdDevGetKernel(i/2,frame_window.getLength()))
      
    if frame_window.isFrameAtEdges() != 0: #if near begin or end of video

      all_kernels = rearrangeGaussianKernels(all_kernels,
                                         frame_window.isFrameAtEdges())
   

    kernel_dict = dict(zip(kernel_keys,all_kernels))
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
    
def rearrangeGaussianKernels(all_kernels, distance_off_center):

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
#TODO: make this less hard-coded --i did and found it less clear
def makeWeightsArray(filter_keys, weights_list):
  """Takes a numpy array of equal dimension to the pixel lum array filled with values of approximately
  how many pixels need to be combined at each pixel (filter_keys).  Associates these with
  the correct elements in weights_list which holds the gaussian weights for the
  different filter_keys.  Will return a numpy array of pixel lum size with values of spatial gaussian weights"""
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
  
  #OR IS IT BETTER NOT TO RETURN TO SHOW MODDED IN PLACE?
  return filter_keys

def nearestFilterKeys(target_nums):
  """Keys in the filter dict are at 1, 1.5, 2, etc..."""
  return np.round(target_nums * 2) / 2
  
 
class FrameQueue(object):
  """Surrounding frame count is the number of frames counting itself.
   Probably need a diff number of surrounding frames for each frame  but
   the best thing to do is probably just overestimate and use less if need be"""
  def __init__(self, video_filename, surrounding_frame_count):
    #position of current frame in window
    self.current_frame_index = 0
    #overall frame number
    self.current_frame = 1    
    self.frame_window = []
    #window is always odd
    if surrounding_frame_count % 2 == 0:
      surrounding_frame_count += 1

    self.frames_in_window = surrounding_frame_coun

    self.frames_in_video = self.countFrames(video_filename)

    if surrounding_frame_count > self.frames_in_video:
      surrounding_frame_count = self.frames_in_video

    self.frames_in_window = surrounding_frame_count

    self.video_capt = cv2.VideoCapture(video_filename)


    self.fourcc = int(self.video_capt.get(cv2.cv.CV_CAP_PROP_FOURCC))
    self.fps =  self.video_capt.get(cv2.cv.CV_CAP_PROP_FPS)
    self.size = (int(self.video_capt.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), \
                   int(self.video_capt.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fc = int(self.video_capt.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    ctr = self.frames_in_window
#then in other method we will be keeping frames at max surr frame count
    while ctr > 0:

      success,image = self.readVidFrameConvertBGR2YUV()
      self.frame_window.append(image)
      ctr -= 1

  def writeVidFrameTEST(self,img,filename):
    image = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    cv2.imwrite(filename, image)

  def writeVidFrameConvertYUV2BGR(self,img,videowriter):
    image = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    videowriter.write(image)

  def readVidFrameConvertBGR2YUV(self):
    success, img = self.video_capt.read()
    if success:
      image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
      return success, image
    else:
      return success,img


  """Use this instead of prop so that videos without metadata can have a dorrect count"""
  def countFrames(self,video_filename):
    capt = cv2.VideoCapture(video_filename)
    if not capt.isOpened():
      raise ValueError("Invalid input file")
    cnt = 0
    success,image = capt.read()
    while success:
      cnt += 1
      success,image = capt.read()
    return cnt
    
  """This returns a window of frames around the current one.  THe only logic comes
  in the beginning and the end when we have to communicate that the current frmae
  is not in the middle of the window"""      
  def getNextFrame(self):
    print "\nWHERE IM LOOKING",self.current_frame
    print "TOTAL", self.frames_in_video
    if self.current_frame > self.frames_in_video:
      return None
    half_window = self.frames_in_window // 2 + 1

    if self.current_frame <= half_window:
      self.current_frame_index += 1

    #advance if out from the beginning and still frames left
    if self.current_frame > half_window and self.current_frame <= (self.frames_in_video - (half_window-1)):

         success,image = self.readVidFrameConvertBGR2YUV()
         #FIFO OP IN NEXT TWO LINES
         self.frame_window.append(image)
         self.frame_window.pop(0)

    if self.current_frame > (self.frames_in_video - (half_window - 1)):

      self.current_frame_index += 1

    #THIS LINE MUST BE RIGHT BEFORE RETURN STATEMENT SO I DONT MESS UP LOGIC
    self.current_frame +=1
    return FrameWindow(self.frame_window,self.current_frame_index)


"""This is the window around the central frame"""
class FrameWindow(object):
  """Given a list of all of its frames at the beginning and this can never be changed. curr_frame_index is 1-indexed rather than 0-indexed"""
  def __init__(self, frame_list,curr_frame_index):

    self.frame_list = frame_list
    self.curr_frame_index = curr_frame_index - 1
    print "CURRENT INDEX!!!!!", self.curr_frame_index
    print "LENGTH", self.getLength()
    print "EDGE INDEX", self.isFrameAtEdges()

  def getMainFrame(self):

    return self.frame_list[self.curr_frame_index]

  def getLength(self):
    return len(self.frame_list)

  def getOtherFrames(self):
    return self.frame_list[0:self.curr_frame_index] + \
           self.frame_list[self.curr_frame_index + 1:]

  def isFrameAtEdges(self):
    """Returns an integer indicating if the frame is close to the beginning or
    ending of the video.  If close to the beginning, it will return a negative
    number indicating how far the center frame is offset from the middle of the
    frame window.  If close to the end, it will return a positive number 
    indicating how far the frame is offset from center"""
    middle_frame_index = self.getLength() // 2

    if middle_frame_index == self.curr_frame_index:
      return 0

    elif self.curr_frame_index < middle_frame_index:
      return self.curr_frame_index - middle_frame_index

    else:
      return self.curr_frame_index - middle_frame_index
    
if __name__ == "__main__":

  faulthandler.enable()
  try:
    frame_queue = FrameQueue('bgu.MOV',19)
  except ValueError as err:
    sys.stderr.write("Invalid Input File\n")
    sys.exit()

  vid = cv2.VideoWriter('new.avi', cv2.cv.CV_FOURCC('M','J','P','G'),
                                  frame_queue.fps,frame_queue.size)


  fw = frame_queue.getNextFrame()


  while fw:
    gain_ratios = findTargetLuminances(fw.getMainFrame())
    result = astaFilter(fw,gain_ratios)
    result = tonemapSpatiallyUniform(result)
    frame_queue.writeVidFrameConvertYUV2BGR(result,vid)
    fw = frame_queue.getNextFrame()

