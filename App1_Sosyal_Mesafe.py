from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations

def yakin(p1, p2):
    
# Amaç1: İki nokta arasındaki Öklid Mesafesini hesaplama

    
    
    dst=math.sqrt(p1**2 + p2**2)
    return dst

def geriDonustur(x, y, w, h):
    
    # Amaç2: Merkez koordinatları dikdörtgen koordinatlara dönüştürür
    """
    bbox: Sınırlayıcı kutuların kullanımını kolaylaştırmayı amaçlayan bir Python kütüphanesi.
        : Param:
     x, y = bbox'ın orta noktası
     w, h = genişlik, bbox yüksekliği
    
     :return:
     xmin, ymin, xmax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvCizimKutu(tespitler, img):

    """
    : Param:
    tespitler = bir çerçevedeki toplam tespitler
    img = darknet'in det_image yönteminden görüntü

    :return:
    bbox ile img
    """
    #Amac 3.1: Kişiler sınıfını tespitlerden filtreleme ve 
    #Her kişi algılama için sınırlayıcı kutu ağırlık merkezi.
    if len(tespitler) > 0:						# Görüntüde en az 1 algılama ve bir çerçevede algılama varlığını kontrol etme  
        agirlikMerkezi_Sozluk = dict() 			# Fonksiyon sözlük oluşturur ve agirlikMerkezi_Sozluk olarak adlandırır
        nesneId = 0								# Nesne ID adlı bir değişkeni başlatıyoruz ve 0 olarak ayarlıyoruz
        for tespit in tespitler:				# Bu if ifadesinde, tüm tespitleri yalnızca kişiler için filtreliyoruz
             # Tek kişinin ad etiketini kontrol edelim
            isim_Etiketi = str(tespit[0].decode())  # Coco dosyası tüm isimlerin dizesini içeriyor
            if isim_Etiketi == 'person':                
                x, y, w, h = tespit[2][0],\
                            tespit[2][1],\
                            tespit[2][2],\
                            tespit[2][3]      	# Algılamaların merkez noktalarını sakla
                xmin, ymin, xmax, ymax = geriDonustur(float(x), float(y), float(w), float(h))   # Merkez koordinatlardan dikdörtgen koordinatlara dönüştürün, BBox'un hassasiyetini sağlamak için float kullanıyoruz
                # Tespit edilen kişiler için bbox'ın merkez noktasını ekleyelim.
                agirlikMerkezi_Sozluk[nesneId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Indeks merkezi noktaları ve bbox olarak 'nesneId' ile tuple sözlüğü oluşturun
                nesneId += 1 #Her algılama için dizini arttır
        #=================================================================#
        #Amac 3.2:Hangi kişilerin bbox'ının birbirine yakın olduğunu belirleme
        
       	
        kirmizi_bolge_list = [] # Hangi Nesne kimliğinin eşik mesafesi koşulunda olduğunu içeren liste. 
        kirmizi_cizgi_list = []
        for (id1, p1), (id2, p2) in combinations(agirlikMerkezi_Sozluk.items(), 2): # Tüm yakın algılama kombinasyonlarını alalım, # Birden fazla öğenin listesi - id1 1, noktalar 2, 1,3
            dx, dy = p1[0] - p2[0], p1[1] - p2[1] 	# Ağırlık merkezi ile x: 0, y: 1 arasındaki farkı kontrol edin
            mesafe = yakin(dx, dy) #Öklid mesafesini hesaplar
            if mesafe < 50.0:						# Sosyal mesafe eşiğimizi belirleyin - Eğer bu koşula uyuyorlarsa ..
                if id1 not in kirmizi_bolge_list:
                    kirmizi_bolge_list.append(id1)       # Listeye kimlik ekle
                    kirmizi_cizgi_list.append(p1[0:2])   # Listeye puan ekleme
                if id2 not in kirmizi_bolge_list:
                    kirmizi_bolge_list.append(id2)		# İkinci kimlik için aynı
                    kirmizi_cizgi_list.append(p2[0:2])
        
        for idx, box in agirlikMerkezi_Sozluk.items():  # Sözlük (1 (anahtar): kırmızı (değer), 2 mavi) idx - anahtar kutusu - değer
            if idx in kirmizi_bolge_list:   # id kırmızı bölge listesindeyse
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 1) # Kırmızı sınırlama kutuları oluşturma # başlangıç noktası, bitiş noktası boyutu 2
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # Yeşil sınırlayıcı kutular oluşturma
		#=================================================================#
    	# Amaç 3.3: Risk Analitiğini Göster ve Risk Göstergelerini Göster  
        text = "Risk Altinda Olan Kisiler: %s" % str(len(kirmizi_bolge_list)) 			# Risk Altındaki İnsanları Sayın
        location = (10,25)												# Görüntülenen metnin konumunu ayarlama
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  # Ekran Metni

        for kontrol in range(0, len(kirmizi_cizgi_list)-1):					# Kırmızı liste öğeleri arasında yinelenen yakındaki kutular arasında çizgi çizme
            baslangic_noktasi = kirmizi_cizgi_list[kontrol] 
            bitis_noktasi = kirmizi_cizgi_list[kontrol+1]
            kontrol_cizgisi_x = abs(bitis_noktasi[0] - baslangic_noktasi[0])   		# X için çizgi koordinatlarını hesaplama
            kontrol_cizgisi_y = abs(bitis_noktasi[1] - baslangic_noktasi[1])			# Y için çizgi koordinatlarını hesaplama
            if (kontrol_cizgisi_x < 50) and (kontrol_cizgisi_y < 25):				# Her ikisi de ise Hatların eşik mesafemizin altında olup olmadığını kontrol ederiz.
                cv2.line(img, baslangic_noktasi, bitis_noktasi, (255, 0, 0), 2)   # Sadece eşik çizgilerinin üzerinde görüntülenir.
        #=================================================================#
    return img


netMain = None
metaMain = None
altNames = None

def YOLO():
    """
    Nesne Tespiti Gerçekleştirimi
    """
    global metaMain, netMain, altNames
    yapilandirmaYolu = "./cfg/yolov3.cfg"
    agirlikYolu = "./yolov3.weights"
    veriYolu = "./cfg/coco.data"
    if not os.path.exists(yapilandirmaYolu):
        raise ValueError("Geçersiz yapılandırma yolu `" +
                         os.path.abspath(yapilandirmaYolu)+"`")
    if not os.path.exists(agirlikYolu):
        raise ValueError("Geçersiz ağırlık yolu `" +
                         os.path.abspath(agirlikYolu)+"`")
    if not os.path.exists(veriYolu):
        raise ValueError("Geçersiz veri dosyası yolu `" +
                         os.path.abspath(veriYolu)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(yapilandirmaYolu.encode(
            "ascii"), agirlikYolu.encode("ascii"), 0, 1)  # yığın boyutu = 1
    if metaMain is None:
        metaMain = darknet.load_meta(veriYolu.encode("ascii"))
    if altNames is None:
        try:
            with open(veriYolu) as metaFH:
                VeriIcinde = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", VeriIcinde,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./Input/test2.mp4")
    cerceve_genisligi = int(cap.get(3))
    cerceve_yuksekligi = int(cap.get(4))
    yeni_yukseklik, yeni_genislik = cerceve_yuksekligi // 2, cerceve_genisligi // 2
    # print("Video Reolution: ",(width, height))

    out = cv2.VideoWriter(
            "./Demo/test5_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
            (yeni_genislik, yeni_yukseklik))
    
    # print("YOLO döngüsü baslatiliyor...")

  # Her tespit için yeniden kullandığımız bir resim oluşturun
    darknet_image = darknet.make_image(yeni_genislik, yeni_yukseklik, 3)
    
    while True:
        onceki_zaman = time.time()
        ret, cerceve_oku = cap.read()
        # Çerçeve mevcut :: 'ret' kare varsa Doğru değerini döndürüp döndürmediğini kontrol edin, aksi takdirde döngüyü kesin.
        if not ret:
            break

        cerceve_rgb = cv2.cvtColor(cerceve_oku, cv2.COLOR_BGR2RGB)
        cerceve_boyut = cv2.resize(cerceve_rgb,
                                   (yeni_genislik, yeni_yukseklik),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,cerceve_boyut.tobytes())

        tespitler = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvCizimKutu(tespitler, cerceve_boyut)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-onceki_zaman))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        out.write(image)

    cap.release()
    out.release()
    print(":::Video yazımı tamamlandı")

if __name__ == "__main__":
    YOLO()