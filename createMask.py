# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:28:13 2017

@author: kcchi
"""

from drawMask import draw_mask
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import maskImage
import scipy.sparse.linalg as scilinalg
import scipy
from PIL import Image
from skimage.io import imsave
import argparse
import pdb

def main(image_file):
    image_file = image_file
    file_name = image_file.split('.')[0].split('_')[0]
    img = Image.open(image_file).convert('RGB')
    img = np.array(img)
    output_name = file_name
    mask = maskImage.maskImage(img, output_name)
    return mask

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", type=str, default=0, help="image path")
	opt = parser.parse_args()
	main(image_file = opt.path)
  


