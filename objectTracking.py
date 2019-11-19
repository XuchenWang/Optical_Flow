import skvideo.io
from boundingBox import boundingBox
import scipy.misc
from getFeatures import getFeatures
from helper import rgb2gray
from estimateAllTranslation import estimateAllTranslation
import cv2
import numpy as np
import matplotlib.pyplot as plt
from applyGeometricTransformation import applyGeometricTransformation


def objectTracking(rawVideo):
    videodata = skvideo.io.vread(rawVideo)
    videodata = videodata[:2,:,:,:] # ONLY for testing, need to be comment out
    scipy.misc.imsave('firstFrame.jpg', videodata[0,:,:,:])
    num_of_box = 1
    expected_feat_per_box = 8

    #for first frame
    img1_gray = rgb2gray(videodata[0,:,:,:])
    coor_matrix = boundingBox('firstFrame.jpg',num_of_box) #get x,y,w,h for bounding box
    feat_x, feat_y = getFeatures(img1_gray,coor_matrix,expected_feat_per_box) #get x,y for feature points
    #     #get color for points
    # scaled_itera = np.array(range(len(feat_x)))/(float(len(feat_x))-1)
    # colors = plt.cm.coolwarm(scaled_itera)
    # print(colors[:3])
    # img1 = np.array(plt.scatter(feat_x, feat_y, color=colors))
    feat_x_flatten = feat_x.flatten()
    feat_y_flatten = feat_y.flatten()
    videodata[0,:,:,:][feat_y_flatten, feat_x_flatten] = [255,0,0]
    x = coor_matrix[:,0]
    y = coor_matrix[:,1]
    w = coor_matrix[:,2]
    h = coor_matrix[:,3]
    upperleft_corner=(x,y)
    lowerright_corner = (x+w,y+h)
    videodata[0,:,:,:] = cv2.rectangle(videodata[0,:,:,:], upperleft_corner, lowerright_corner, (255, 0, 0), 2)
    plt.imshow(videodata[0,:,:,:])
    plt.show()


    for i in range(1,videodata.shape[0]):
        img2_gray = rgb2gray(videodata[i,:,:,:])
        new_feat_x, new_feat_y = estimateAllTranslation(feat_x,feat_y,img1_gray,img2_gray)
        new_feat_x, new_feat_y, new_coor_matrix = applyGeometricTransformation(\
            feat_x, feat_y, new_feat_x, new_feat_y, coor_matrix)
        #plot feature points
        feat_x_flatten = np.vstack((feat_x_flatten,new_feat_x.flatten()))
        feat_y_flatten = np.vstack((feat_y_flatten,new_feat_y.flatten()))
        videodata[i,:,:,:][feat_y_flatten, feat_x_flatten] = [255,0,0]
        #plot bounding box
        x = new_coor_matrix[:,0]
        y = new_coor_matrix[:,1]
        w = new_coor_matrix[:,2]
        h = new_coor_matrix[:,3]
        upperleft_corner=(x,y)
        lowerright_corner = (x+w,y+h)
        videodata[i,:,:,:] = cv2.rectangle(videodata[i,:,:,:], upperleft_corner, lowerright_corner, (255, 0, 0), 2)
        plt.imshow(videodata[i,:,:,:])
        plt.show()

        #updating
        img1_gray = img2_gray
        coor_matrix = new_coor_matrix
        feat_x, feat_y = new_feat_x, new_feat_y



if __name__== '__main__':
    file_name = 'Easy.mp4'
    objectTracking(file_name)

