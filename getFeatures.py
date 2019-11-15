"""
[x,y]=getFeatures(img,bbox)
• (INPUT) img: H × W matrix representing the grayscale input image
• (INPUT) bbox: F × 4 × 2 matrix representing the four corners of the bounding box where F is the number of objects you would like to track
• (OUTPUT) x: N × F matrix representing the N row coordinates of the features across F objects
• (OUTPUT) y: N × F matrix representing the N column coordinates of the features across F objects
"""

from cv2 import goodFeaturesToTrack
import numpy as np
from cv2 import cornerHarris
from anms import anms

# img: H × W matrix representing the grayscale input image
# bbox: F × 4 matrix representing the four corners of the bounding box: x,y,w,h
# N is the expecting number of features in each object box
def getFeatures(img, bbox, N):
    F = bbox.shape[0]
    return_x = np.zeros([N,F])
    return_y = np.zeros([N,F])

    for i in range(F):
        xywh = bbox[i,:]
        x = xywh[0]
        y = xywh[1]
        w = xywh[2]
        h = xywh[3]
        boxed_image = img[y:y+h, x:x+w]

        # alternatively ucing cv2.goodFeaturesToTrack for corner detector and anms
        corners = goodFeaturesToTrack(boxed_image,N,0.01,10)
        corners = np.int0(corners)
        feat_x = corners[:,0,0]
        feat_y = corners[:,0,1]
        feat_y = feat_y+y
        feat_x = feat_x+x

        # #using cornerHarris and anms
        # cimg = cornerHarris(boxed_image, 10, 3, 0.001)
        # cimg[cimg < np.max(cimg)*0.01] = 0
        # feat_y,feat_x = np.nonzero(cimg)
        # feat_y = feat_y+y
        # feat_x = feat_x+x
        # curr_n = len(feat_x)
        # if curr_n>N:
        #     feat_x, feat_y, rmax = anms(cimg, N)
        # else:
        #     feat_x = np.concatenate((feat_x, np.array([0]*(N-curr_n))))
        #     feat_y = np.concatenate((feat_y, np.array([0]*(N-curr_n))))
        return_x[:,i] = feat_x
        return_y[:,i] = feat_y

    return return_x,return_y

