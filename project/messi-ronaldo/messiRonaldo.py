import cv2
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
resim = cv2.imread('messi ve ronaldo.jpg')
griResim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
rects = detector.detectMultiScale(griResim, scaleFactor=1.25, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
k = 0 # tespit edilen yüz sayısı
isim = ['Messi', 'Ronaldo'] # oyuncuların isimlerini dikdörtgenin sol üst köşesinden x ekseninde kaydır
isimKaydirX = [80, 85]
for (x, y, w, h) in rects: # tespit edilen yüzleri dikdörtgen içine al ve oyuncunun ismini yazdır
    cv2.rectangle(resim, (x, y), (x + w, y + h), (0, 255, 255), 2)
    resim = cv2.putText(resim, isim[k], (x-isimKaydirX[k], y+int(h/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
    k = k + 1 # tespit edilen yüz varsa artır
yazi0 = 'Resimde %i tane yuz tespit edildi' %k
resim = cv2.putText(resim, yazi0, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
yazi1 = 'Yukseklik = %i piksel' %resim.shape[0]
yazi2 = 'Genislik = %i piksel' %resim.shape[1]
resim = cv2.putText(resim,yazi1,(550, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 1, cv2.LINE_AA)
resim = cv2.putText(resim,yazi2,(550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 1, cv2.LINE_AA)
cv2.imshow('yuz tespiti', resim)
cv2.waitKey(0) # kullanıcı bir tuşa basana kadar ekranda görüntüle
cv2.imwrite('ronaldo and messi annotated.jpg', resim, [cv2.IMWRITE_JPEG_QUALITY, 100])