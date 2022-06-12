# Written By Onur Yetim
import cv2
import numpy as np

print('[BİLGİ] Yüz Tespiti için Haar Cascade yükleniyor...')
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("[BİLGİ] Yüz tespiti için derin öğrenme modeli olan ResNet Caffe'den yükleniyor...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
conf = 0.5 # minimum probability to filter weak detections
k = 0 

for i in range(47,114): # furkan-data-1 için 66, furkan-data-2 için 47,114,1
    img1 = cv2.imread('onur-data-2/left cam %i.jpg' %i)
    img2 = cv2.imread('onur-data-2/right cam %i.jpg' %i)
    img3 = cv2.imread('onur-data-2/right cam %i.jpg' %i)
    img4 = cv2.imread('onur-data-2/right cam %i.jpg' %i)
    k +=1
    (h,w,c) = img1.shape
    stereo = np.zeros((2*h,2*w,c), np.uint8)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    rects1 = detector.detectMultiScale(gray1, scaleFactor=1.05, minNeighbors=25)
    rects2 = detector.detectMultiScale(gray2, scaleFactor=1.05, minNeighbors=25)
    
    img1 = cv2.putText(img1, 'Left Cam' , (480,40), 0, 1, (0,255,0), 2, 0)
    img2 = cv2.putText(img2, 'Right Cam', (460,40), 0, 1, (0,255,0), 2, 0)
    img3 = cv2.putText(img3, 'Left Cam' , (480,40), 0, 1, (0,0,255), 2, 0)
    img4 = cv2.putText(img4, 'Right Cam', (460,40), 0, 1, (0,0,255), 2, 0)
    img1 = cv2.putText(img1, 'Frame #%i'  %(k), (30,40), 0, 1, (0,255,0), 2, 0)
    img2 = cv2.putText(img2, 'Frame #%i'  %(k), (30,40), 0, 1, (0,255,0), 2, 0)
    img3 = cv2.putText(img3, 'Frame #%i'  %(k), (30,40), 0, 1, (0,0,255), 2, 0)
    img4 = cv2.putText(img4, 'Frame #%i'  %(k), (30,40), 0, 1, (0,0,255), 2, 0)
    #
    blob1 = cv2.dnn.blobFromImage(cv2.resize(img3, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
    blob2 = cv2.dnn.blobFromImage(cv2.resize(img4, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the detections and predictions
    net.setInput(blob1)
    detections1 = net.forward()
    net.setInput(blob2)
    detections2 = net.forward()
    cv2.putText(img3, 'Tespit edilen yuz sayisi: %i' %len(detections1), (20,460), 2, 0.6, (0,0,255), 1)
    cv2.putText(img4, 'Tespit edilen yuz sayisi: %i' %len(detections2), (20,460), 2, 0.6, (0,0,255), 1)
    
    # tespit edilen yüzleri dikdörtgen olarak çiz ve piksel koordinatlarını yaz
    for (x,y,width,height) in rects1:
        cv2.rectangle(img1, (x,y), (x+width,y+height), (0,255,0), 2)
        cv2.putText(img1,'(%.2f,%.2f)' %(x+width/2,y+height/2),(x-30,y-10),2,0.45,(0,255,0),1)
    cv2.putText(img1, 'Tespit edilen yuz sayisi: %i' %len(rects1), (20,460), 2, 0.6, (0,255,0), 1)
    for (x,y,width,height) in rects2:
        cv2.rectangle(img2, (x,y), (x+width,y+height), (0,255,0), 2)
        cv2.putText(img2, '(%.2f,%.2f)' %(x+width/2,y+height/2), (x-30,y-10), 2, 0.45, (0,255,0), 1)
    cv2.putText(img2, 'Tespit edilen yuz sayisi: %i' %len(rects2), (20,460), 2, 0.6, (0,255,0), 1)
   
    
    
    # loop over the detections
    # loop over the detections
    for i in range(0, detections1.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
        confidence = detections1[0, 0, i, 2]
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if confidence < conf:
            continue
		# compute the (x,y)-coordinates of the bounding box for the object
        box = detections1[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
		# draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img3, (startX,startY), (endX,endY), (0, 0, 255), 2)
        cv2.putText(img3, text, (startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
    for i in range(0, detections2.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
        confidence = detections2[0, 0, i, 2]
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if confidence < conf:
            continue
		# compute the (x,y)-coordinates of the bounding box for the object
        box = detections2[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
		# draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img4, (startX,startY), (endX,endY), (0, 0, 255), 2)
        cv2.putText(img4, text, (startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)  

         
    # resmi göster
    stereo[0:h,0:w,:] = img1
    stereo[0:h,w:2*w,:] = img2
    stereo[h:2*h,0:w,:] = img3
    stereo[h:2*h,w:2*w,:] = img4
    s = 0.6
    rstereo = cv2.putText(stereo, 'Haar Cascade Face Detector', (420,80), 0, 1, (0,255,0), 2, 0)
    rstereo = cv2.putText(stereo, 'ResNet DL Face Detector (Caffe)', (380,580), 0, 1, (0,0,255), 2, 0)
    rstereo = cv2.putText(stereo, 'SAMPIYON ', (480,680), 0, 1, (255,0,0), 2, 0)
    rstereo = cv2.putText(stereo, 'FENERBAHCE ',  (650,680), 0, 1, (0,255,255), 2, 0)
    rstereo = cv2.resize(stereo, (int(s*stereo.shape[1]), int(s*stereo.shape[0])), cv2.INTER_LINEAR)
    cv2.imshow('Haar cascade metodu ve Derin ogrenme ile yuz tespiti ', rstereo) 
    if cv2.waitKey(1) == 27: # ESC'ye basınca çık
        break
cv2.destroyAllWindows()