import cv2
resim = cv2.imread('karasu.jpg') # imread komutu , resimi okur
print('yükseklik = %i   genişlik = %i   kanal sayısı = %i' %(resim.shape[0],resim.shape[1],resim.shape[2])) 
# resmin üzerine yazı yazalım
font1 = cv2.FONT_HERSHEY_SIMPLEX # font tipi
org1 = (80, 120) # yazının içinde bulunduğu dikdörtgenin sol alt köşesi
fontScale1 = 3 # font büyüklüğü
color1 = (0, 0, 0) # BGR sırasında yazının renk kodu
thickness1 = 6 # yazının kalınlığı
font2 = cv2.FONT_HERSHEY_SIMPLEX # font tipi
org2 = (480, 120) # yazının içinde bulunduğu dikdörtgenin sol alt köşesi
fontScale2 = 3 # font büyüklüğü
color2 = (0, 255, 0) # BGR sırasında yazının renk kodu
thickness2 = 6 # yazının kalınlığı
yaziliResim = cv2.putText(resim, 'Sakarya', org1, font1, fontScale1, color1, thickness1, cv2.LINE_AA)
yaziliResim = cv2.putText(resim, 'Karasu', org2, font2, fontScale2, color2, thickness2, cv2.LINE_AA)
# resmi yeniden boyulandır, dosyaya kaydet ve ekranda görüntüle
s = 0.8 # scale - ölçek
dim = (int(s*resim.shape[1]), int(s*resim.shape[0])) # boyut
yeniYaziliResim = cv2.resize(yaziliResim, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite('karasu2.jpg', yeniYaziliResim, [cv2.IMWRITE_JPEG_QUALITY, 100]) # imwrite, yeni oluşturulan(ramden harddiske)resmi kaydet. Qualıty, Resmin Kalitesi
cv2.imshow("Uzerine yazi yazilmis ve yeniden boyutlandirilmis resim", yeniYaziliResim) # imshow ekranda görüntüleme
cv2.waitKey(0) #klavyede herhangi  bir tusa basana kadar bekle