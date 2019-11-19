import numpy as np
from skimage import transform as trans

def xywh_to_ptsxy(bbox):
    bbox_form = np.zeros((bbox.shape[0],4,2))
    for i in range(bbox.shape[0]):
        x,y,w,h = bbox[i,:]
        corner_left_up = (x,y)
        corner_right_up = (x+w,y)
        corner_left_down = (x,y+h)
        corner_right_down = (x+w,y+h)

        bbox_form[i,0,:] = corner_left_up
        bbox_form[i,1,:] = corner_right_up
        bbox_form[i,2,:] = corner_left_down
        bbox_form[i,3,:] = corner_right_down
    return bbox_form

def ptsxy_to_xywh(bbox_form):
    bbox = np.zeros((bbox_form.shape[0],4))
    for i in range(bbox_form.shape[0]):
        corner_coor = bbox_form[i,:,:]
        (x,y) = corner_coor[0,:]
        w = corner_coor[1,0] - corner_coor[0,0]
        h = corner_coor[2,1] - corner_coor[0,1]
        bbox[i,:] = (x,y,w,h)
    return bbox

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    N, F = startXs.shape
    bbox_form = xywh_to_ptsxy(bbox)
    _, window_corner_num, dim_num = bbox_form.shape
    # First we will filter the points whose distance is greater than
    dist_mat = np.sqrt((newXs-startXs)**2+(newYs-startYs)**2)
    filter_index = np.where(dist_mat > 4)

    # filtered new points
    filter_newXs = newXs.copy()
    filter_newXs[filter_index] = -1

    filter_newYs = newYs.copy()
    filter_newYs[filter_index] = -1

    # filtered start points
    filter_startXs = startXs.copy()
    filter_startXs[filter_index] = -1

    filter_startYs = startYs.copy()
    filter_startYs[filter_index] = -1

    # compulate the Similar matrix H for each window
    newbbox_form = np.zeros((F,window_corner_num,dim_num))
    Xs = filter_newXs.copy()
    Ys = filter_newYs.copy()
    for i in range(F):
        # for each window, filter out the out-ranged pts
        src_pts_x = filter_startXs[:, i]
        src_pts_x = src_pts_x[src_pts_x>=0].reshape(-1,1)
        src_pts_y = filter_startYs[:, i]
        src_pts_y = src_pts_y[src_pts_y >= 0].reshape(-1,1)

        tar_pts_x = filter_newXs[:, i]
        tar_pts_x = tar_pts_x[tar_pts_x >= 0].reshape(-1,1)
        tar_pts_y = filter_newYs[:, i]
        tar_pts_y = tar_pts_y[tar_pts_y >= 0].reshape(-1,1)

        src = np.hstack((src_pts_x,src_pts_y))
        tar = np.hstack((tar_pts_x,tar_pts_y))

        tform = trans.SimilarityTransform()
        res = tform.estimate(tar, src)
        H_matrix_i = tform.params
        bbox_form_i = bbox_form[i,:,:]
        aug_ones = np.ones(window_corner_num)
        aug_bbox_i = np.vstack((bbox_form_i.T, aug_ones))

        newbbox_i = H_matrix_i@aug_bbox_i
        newbbox_i = (newbbox_i/newbbox_i[2,:])[:2,:]

        newbbox_form[i,:,:] = newbbox_i.T

        bbox_i_x_min = np.min(newbbox_i[0])
        bbox_i_x_max = np.max(newbbox_i[0])
        bbox_i_y_min = np.min(newbbox_i[1])
        bbox_i_y_max = np.max(newbbox_i[1])

        # as long as one of the coor is out, the point is out
        logic_x = np.logical_or(np.where(tar_pts_x>bbox_i_x_max),\
                                np.where(tar_pts_x<bbox_i_x_min))
        logic_y = np.logical_or(np.where(tar_pts_y > bbox_i_y_max), \
                                np.where(tar_pts_y < bbox_i_y_min))
        logic = np.logical_or(logic_x,logic_y)

        Xs_i = tar_pts_x.copy()
        Ys_i = tar_pts_y.copy()
        if logic != []:
            Xs_i[logic] = -1
            Ys_i[logic] = -1

        Xs[:,i] = Xs_i[:,0]
        Ys[:,i] = Ys_i[:,0]
    newbbox = ptsxy_to_xywh(newbbox_form)

    return Xs, Ys, newbbox
