import skvideo.io
from boundingBox import boundingBox

def objectTracking(rawVideo):
    videodata = skvideo.io.vread(rawVideo)
    box_corner = boundingBox(videodata[0,:,:,:])




if __name__== '__main__':
    file_name = 'Easy.mp4'
    objectTracking(file_name)

