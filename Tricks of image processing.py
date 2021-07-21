# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 22:17:21 2021

@author: abc
"""

#image colorization using pillow library
from PIL import Image, ImageOps

#read our image and convert it into gray level image
img = Image.open(r"BSE_Image.jpg").convert("L")

#colorize our image
img1 = ImageOps.colorize(img, black="blue", white="red")

#show our image
img1.show()


########################################################

#Image colorization using opencv
import cv2

#read our image
gray_img = cv2.imread("BSE_Image.jpg")

#Colorize our image
color_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)

#SHOW OUR IMAGE
cv2.imshow("Original image", gray_img)
cv2.imshow("Color image", color_img)
cv2.waitKey()
cv2.destroyALLWindows()




#####################################################

#Denoise image using gaussian filter 
from skimage import img_as_float
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd

#read our image as float
img = img_as_float(io.imread("BSE_Google_noisy.jpg"))
#need to convert to float as we will be doing math on the array

#Apply gaussian filter 
gaussian_img = nd.gaussian_filter(img, sigma=3)

#show the image
plt.imshow(gaussian_img,cmap="gray")



############################################################


#Denoise image using Anisotropic diffusion
import matplotlib.pyplot as plt
import cv2
from skimage import io
from medpy.filter.smoothing import anisotropic_diffusion

#read our image 
img = io.imread("BSE_Google_noisy.jpg", as_gray=True)

#Apply anisotropic diffusion filter on our image
img_filtered = anisotropic_diffusion(img, niter=5, kappa=50, gamma=0.1, option=2)
plt.imshow(img_filtered, cmap = "gray")
plt.imsave("anisotropic_result.jpg",img_filtered, cmap="gray")


###############################################################################


#Segmentation 
import matplotlib.pyplot as plt
from skimage import io














