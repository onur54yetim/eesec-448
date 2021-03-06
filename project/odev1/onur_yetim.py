import cv2 

foto_zurafa = cv2.imread('giraffe.jpg',cv2.IMREAD_COLOR)

ffoto_onur1 = cv2.blur(foto_zurafa, [20, 20])
ffoto_onur2 = cv2.GaussianBlur(foto_zurafa, [15, 15], 0)
ffoto_onur3 = cv2.medianBlur(foto_zurafa, 45)
ffoto_onur4 = cv2.bilateralFilter(foto_zurafa, 45, 55, 65)
s = 0.2
m = (int(s*foto_zurafa.shape[1]), int(s*foto_zurafa.shape[0]))
foto_zurafa = cv2.resize(foto_zurafa, m, cv2.INTER_LINEAR)
foto_onur1 = cv2.resize(ffoto_onur1, m, cv2.INTER_LINEAR)
foto_onur2 = cv2.resize(ffoto_onur2, m, cv2.INTER_LINEAR)
foto_onur3 = cv2.resize(ffoto_onur3, m, cv2.INTER_LINEAR)
foto_onur4= cv2.resize(ffoto_onur4, m, cv2.INTER_LINEAR)
cv2.imshow('Orjinal Foto',foto_zurafa )
cv2.imshow('Blur Foto',foto_onur1 )
cv2.imshow('GaussianBlur Foto', foto_onur2)
cv2.imshow('MedianBlur Foto', foto_onur3)
cv2.imshow('BilateralFilter Foto ', foto_onur4)
cv2.imwrite('giraffe blur.jpg',foto_onur1 , [cv2.IMWRITE_JPEG_QUALITY, 100])
cv2.imwrite('giraffe GaussianBlur.jpg',foto_onur2 , [cv2.IMWRITE_JPEG_QUALITY, 100])
cv2.imwrite('giraffe medianBlur.jpg',foto_onur3 , [cv2.IMWRITE_JPEG_QUALITY, 100])
cv2.imwrite('giraffe bilateralFilter.jpg',foto_onur4, [cv2.IMWRITE_JPEG_QUALITY, 100])
cv2.waitKey(0) 