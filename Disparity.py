import cv2
import numpy as np

#↓input variables:
input_image_l= cv2.imread('left.jpg', 0)/255
input_image_r= cv2.imread('right.jpg', 0)/255

input_image_l = cv2.resize(input_image_l, (input_image_l.shape[1]*2, input_image_l.shape[0]*2), interpolation=cv2.INTER_CUBIC)
input_image_r = cv2.resize(input_image_r, (input_image_r.shape[1]*2, input_image_r.shape[0]*2), interpolation=cv2.INTER_CUBIC)
bluredLeft = cv2.GaussianBlur(input_image_l, (15,15), 1)
bluredRight = cv2.GaussianBlur(input_image_r, (15,15), 1)
sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
input_image_l = cv2.filter2D(bluredLeft, -1, sharpen)
input_image_r = cv2.filter2D(bluredRight, -1, sharpen)

def computedisparity(left, right):
    #YOU CAN CHANGE WINDOW SIZE HERE:
    windowSize=12
    width = len(left)
    height = len(left[0])
    disp = np.zeros((width,height))
    SSD = np.zeros((width,height,100))
    for i in range(100):
        SSD[:, :, i] = (left-np.roll(right,i))**2
    for i in range(0, height-windowSize):
      for k in range(0, width-windowSize):
        SSDk = SSD[k:(k+windowSize),i:(i+windowSize),:]
        SSDsum = np.sum(np.sum(SSDk,axis=0),axis=0)
        d = SSDsum.argmin()
        disp[k,i] = d
        dShape = int(disp.shape[1]*9/10)
        out = disp[:,0:dShape]
    return cv2.normalize(out,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX).astype(np.uint8)

disparity = computedisparity(input_image_l,input_image_r)
cv2.imwrite('disparity.jpg', disparity)