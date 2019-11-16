import numpy as np
from skimage import transform as trans

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    N, F = startXs.shape
    _, window_corner_num, dim_num = bbox.shape
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
    newbbox = np.zeros((F,window_corner_num,dim_num))
    Xs = filter_newXs.copy()
    Ys = filter_newYs.copy()
    for i in range(F):
        # for each window, filter out the out-ranged pts
        src_pts_x = filter_startXs[:, i]
        src_pts_x = src_pts_x[src_pts_x>=0]
        src_pts_y = filter_startYs[:, i]
        src_pts_y = src_pts_y[src_pts_y >= 0].reshape(-1,1)

        tar_pts_x = filter_newXs[:, i]
        tar_pts_x = tar_pts_x[tar_pts_x >= 0]
        tar_pts_y = filter_newYs[:, i]
        tar_pts_y = tar_pts_y[tar_pts_y >= 0].reshape(-1,1)

        src = np.hstack((src_pts_x,src_pts_y))
        tar = np.hstack((tar_pts_x,tar_pts_y))

        tform = trans.SimilarityTransform()
        H_matrix_i = tform.estimate(src, tar)

        bbox_i = bbox[i,:,:]
        aug_ones = np.ones(window_corner_num)
        aug_bbox_i = np.vstack((bbox_i, aug_ones))

        newbbox_i = H_matrix_i@aug_bbox_i
        newbbox_i = (newbbox_i/newbbox_i[2,:])[:2,:]

        newbbox[i,:,:] = newbbox_i

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
        Xs_i[logic] = -1
        Ys_i = tar_pts_y.copy()
        Ys_i[logic] = -1

        Xs[:,i] = Xs_i
        Ys[:,i] = Ys_i

    return Xs, Ys, newbbox
