import skvideo.io
from boundingBox import boundingBox
import scipy.misc
def objectTracking(rawVideo):
    videodata = skvideo.io.vread(rawVideo)
    scipy.misc.imsave('firstFrame.jpg', videodata[0,:,:,:])
    # box_corner = boundingBox(videodata[0,:,:,:])




if __name__== '__main__':
    file_name = 'Easy.mp4'
    objectTracking(file_name)

