# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 00:02:01 2017

@author: kcchi
"""
from drawMask import draw_mask
import numpy as np
from skimage.io import imsave

def maskImage(img, output_name):

	mask_, bbox_ = draw_mask(img)
	# imsave(str(output_name)+'_mask.png', mask_)

	return mask_


