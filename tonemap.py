from __future__ import division
import cv2
import numpy as np
from math import log10



def toneMap(pixel_lum,attenuation):
  """Takes as argument an input pixel luminance and maps it to an output pixel luminance. """
  ratio = pixel_lum / 255 #255 is max luminance
  num = log10(ratio * (attenuation - 1) + 1)
  denom = log10(attenuation)
  return num/denom


def toneMapVectorized(vidFrame,attenuation):
  lum = vidFrame[:,:,0]
  toneMapVectorFunction = np.vectorize(toneMap)
  return toneMapVectorFunction(lum,attenuation)
  

def divideIfNonZero(num,denom):
  if denom == 0.0:
    return 1.0
  return num / denom
  
def divideIfNonZeroVec(num_array,denom_array):
  vecDivideNonZero = np.vectorize(divideIfNonZero)
  return vecDivideNonZero(num_array,denom_array)
 
def findTargetLuminances(vidFrame):

  throwaway_copy = np.copy(vidFrame)
  throwaway_blurred = cv2.GaussianBlur(throwaway_copy,(7,7),0)
  original_luminances = np.copy(vidFrame[:,:,0])
  result = toneMapVectorized(throwaway_blurred,34)
  result *= 255
  return divideIfNonZeroVec(result,throwaway_blurred[:,:,0])

def tonemapSpatiallyUniform(vidFrame):
  result = toneMapVectorized(vidFrame,34)
  result *= 255
  vidFrame[:,:,0] = result.astype(int)
  return vidFrame

if __name__ == "__main__":

  
  for i in xrange(0,255):
    print i, " ",toneMap(i,40) * 255
