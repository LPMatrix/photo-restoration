import pandas as pd 
import numpy as np 
import cv2

image = cv2.imread('r3.jpg')
cv2.imshow('Original image', image)
cv2.waitKey(0)

damages = cv2.imread('r3.jpg', 0)
cv2.imshow('Damages', damages)
cv2.waitKey(0)

ret, thresh1 = cv2.threshold(damages, 245, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold', thresh1)
cv2.waitKey(0)

kernel = np.ones((7,7), np.uint8)
dilated = cv2.dilate(thresh1, kernel, iterations=1)
cv2.imshow('Dilated', dilated)
cv2.waitKey(0)

restored = cv2.inpaint(image, dilated, 3, cv2.INPAINT_TELEA)
cv2.imshow('Restored', restored)
cv2.waitKey(0)

cv2.destroyAllWindows()