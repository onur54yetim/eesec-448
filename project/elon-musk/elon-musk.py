import cv2
import numpy as np

resim = cv2.imread('Elon_Musk.jpg')
resimGray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
(T,resimBW1) = cv2.threshold(resimGray, 50,  255,  cv2.THRESH_BINARY)
(T,resimBW2) = cv2.threshold(resimGray, 100, 255, cv2.THRESH_BINARY)  
(T,resimBW3) = cv2.threshold(resimGray, 150, 255, cv2.THRESH_BINARY)

(h,w)=resimGray.shape
resim = np.zeros((h,3*w), np.uint8)
resim [:,0:w] = resimBW3
resim [:,w:2*w] =resimBW1
resim [:,2*w:3*w] =resimBW2

resim = cv2.putText(resim, '(1)', (30,50), 1, 1, (0,0,0), 2, 0)
resim= cv2.putText(resim, '(2)', (w+30,50), 1, 1, (0,0,0), 2, 0)
resim= cv2.putText(resim, '(3)', (2*w+30,50), 1, 1, (0,0,0), 2, 0)

s = 0.1
imgGrayResized = cv2.resize(resimGray, (int(s*resimGray.shape[1]/2), int(s*resimGray.shape[0]/2)), 0)


cv2.imwrite('elon musk binary.jpg', resim,[cv2.IMWRITE_JPEG_QUALITY,100])
cv2.imshow('elon musk binary', resim) 
cv2.waitKey(0)              

