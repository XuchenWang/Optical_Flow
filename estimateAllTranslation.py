"""
[newXs, newYs] = estimateAllTranslation(startXs,startYs,img1,img2)
• (INPUT) startXs: N × F matrix representing the starting X coordinates of all the features in the first frame for
all the bounding boxes
• (INPUT) startYs: N × F matrix representing the starting Y coordinates of all the features in the first frame for all the bounding boxes
• (INPUT)img1:H×W×3matrixrepresentingthefirstimageframe
• (INPUT)img2:H×W×3matrixrepresentingthesecondimageframe
• (OUTPUT) newXs: N × F matrix representing the new X coordinates of all the features in all the bounding boxes
• (OUTPUT) newYs: N × F matrix representing the new Y coordinates of all the features in all the bounding boxes
"""


# startXs: N × F
# img1, img2: H×W×3
def estimateAllTranslation(startXs, startYs, img1, img2):
    newXs = startXs.copy()
    newYs = startYs.copy()



    return newXs, newYs
