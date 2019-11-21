import skvideo.io
from boundingBox import boundingBox
from getFeatures import getFeatures
from helper import rgb2gray
from estimateAllTranslation import estimateAllTranslation
import cv2
import numpy as np
import matplotlib.pyplot as plt
from applyGeometricTransformation import applyGeometricTransformation
import imageio


def objectTracking(rawVideo):
    videodata = skvideo.io.vread(rawVideo)
    videodata = videodata[:100,:,:,:] # ONLY for testing, need to be comment out
    # scipy.misc.imsave('firstFrame.jpg', videodata[0,:,:,:])
    imageio.imwrite('firstFrame.jpg', videodata[0,:,:,:])
    num_of_box = 2
    expected_feat_per_box = 10
    windowSize = 10

    #for first frame
    img1_gray = rgb2gray(videodata[0,:,:,:])
    coor_matrix = boundingBox('firstFrame.jpg',num_of_box) #get x,y,w,h for bounding box
    feat_x, feat_y = getFeatures(img1_gray,coor_matrix,expected_feat_per_box) #get x,y for feature points
    feat_x_flatten = feat_x.flatten()
    feat_y_flatten = feat_y.flatten()
    videodata[0,:,:,:][feat_y_flatten, feat_x_flatten] = [255,0,0]
    videodata[0,:,:,:][feat_y_flatten-1, feat_x_flatten] = [255,0,0] #bigger points
    videodata[0,:,:,:][feat_y_flatten+1, feat_x_flatten] = [255,0,0] #bigger points
    videodata[0,:,:,:][feat_y_flatten, feat_x_flatten-1] = [255,0,0] #bigger points
    videodata[0,:,:,:][feat_y_flatten, feat_x_flatten+1] = [255,0,0] #bigger points
    x = coor_matrix[:,0]
    y = coor_matrix[:,1]
    w = coor_matrix[:,2]
    h = coor_matrix[:,3]
    upperleft_corner=np.array([x,y])
    lowerright_corner = np.array([x+w,y+h])
    for i in range(x.shape[0]):
        videodata[0,:,:,:] = cv2.rectangle(videodata[0,:,:,:], tuple(upperleft_corner[:,i]), tuple(lowerright_corner[:,i]), (255, 0, 0), 2)
    # scipy.misc.imsave(str(0)+'thFrame.jpg', videodata[0,:,:,:])
    imageio.imwrite(str(0)+'thFrame.jpg', videodata[0,:,:,:])


    for i in range(1,videodata.shape[0]):
        img2_gray = rgb2gray(videodata[i,:,:,:])
        new_feat_x, new_feat_y = estimateAllTranslation(feat_x,feat_y,img1_gray,img2_gray, windowSize)
        new_feat_x, new_feat_y, new_coor_matrix = applyGeometricTransformation(\
            feat_x, feat_y, new_feat_x, new_feat_y, coor_matrix)
        #plot feature points
        feat_x_flatten = np.vstack((feat_x_flatten,new_feat_x.flatten()))
        feat_y_flatten = np.vstack((feat_y_flatten,new_feat_y.flatten()))
        videodata[i,:,:,:][feat_y_flatten, feat_x_flatten] = [255,0,0]
        videodata[i,:,:,:][feat_y_flatten-1, feat_x_flatten] = [255,0,0] #for bigger points
        videodata[i,:,:,:][feat_y_flatten+1, feat_x_flatten] = [255,0,0]
        videodata[i,:,:,:][feat_y_flatten, feat_x_flatten-1] = [255,0,0]
        videodata[i,:,:,:][feat_y_flatten, feat_x_flatten+1] = [255,0,0]

        #plot bounding box
        x = new_coor_matrix[:,0]
        y = new_coor_matrix[:,1]
        w = new_coor_matrix[:,2]
        h = new_coor_matrix[:,3]
        upperleft_corner=np.array([x,y])
        lowerright_corner = np.array([x+w,y+h])
        for j in range(x.shape[0]):
            videodata[i,:,:,:] = cv2.rectangle(videodata[i,:,:,:], tuple(upperleft_corner[:,j]), tuple(lowerright_corner[:,j]), (255, 0, 0), 2)
        # scipy.misc.imsave(str(i)+'thFrame.jpg', videodata[i,:,:,:])
        imageio.imwrite(str(i)+'thFrame.jpg', videodata[i,:,:,:])
        # plt.imshow(videodata[i,:,:,:])
        # plt.show()

        #updating
        img1_gray = img2_gray
        coor_matrix = new_coor_matrix
        feat_x, feat_y = new_feat_x, new_feat_y

    tracking_list = []
    for i in range(videodata.shape[0]):
        tracking_list.append(videodata[i, :, :, :])
        imageio.mimsave('./eval_tracking.gif', tracking_list)

if __name__== '__main__':
    file_name = 'Easy.mp4'
    objectTracking(file_name)

