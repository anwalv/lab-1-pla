import cv2
import numpy as np
image1 = cv2.imread('photo_2024-06-02_01-45-04.jpg')
image2 = cv2.imread('photo_2024-06-02_01-45-08.jpg')

resized_image = cv2.resize(image2, (400, 700))
cv2.imshow('Resized Image',resized_image)
cv2.waitKey(0)
rotated_image = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Rotated Image',rotated_image)
cv2.waitKey(0)