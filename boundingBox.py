import cv2
import createMask


def boundingBox(firstFrame):
    mask = createMask.main(firstFrame)
    x,y,w,h = cv2.boundingRect(mask) # (x,y) is the top-left coordinate of the rectangle and (w,h) be its width and height.
    cv2.rectangle(mask,(x,y),(x+w,y+h),(0,255,0),2)

    return None
