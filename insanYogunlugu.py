
from ctypes import *
import os
import cv2
import darknet
import glob
import math
import random
import numpy as np
import time
import darknet
from itertools import combinations

def geriDonus(x, y, w, h):								# Merkez koordinatlardan sınırlayıcı kutu koordinatlarına dönüştürme
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvCizimKutu(tespitler, img):

    # Amaç 1: Kişi Sayma
    #================================================================    
    if len(tespitler) > 0:    								# Herhangi bir tespit varsa
        insan_tespit = 0
        for tespit in tespitler:						# Her algılama için
            isim_etiketi = tespit[0].decode()				# Sınıf listesinin kodunu çöz
            if isim_etiketi == 'person':							# Insan sınıfı için filtre algılamaları
	            x, y, w, h = tespit[2][0],\
	                tespit[2][1],\
	                tespit[2][2],\
	                tespit[2][3]  						# Algılama koordinatlarını alma
	            xmin, ymin, xmax, ymax = geriDonus(
	                float(x), float(y), float(w), float(h))  	# Sınırlayıcı kutu koordinatlarına dönüştür
	            pt1 = (xmin, ymin)								# Nokta 1 ve 2 için Biçim Koordinatları
	            pt2 = (xmax, ymax)
	            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)  	# Dikdörtgenlerimizi çizin
            insan_tespit += 1 								# Bir sonraki algılanan kişiye artış
        cv2.putText(img,
        "Insan Yogunlugu %s" % str(insan_tespit), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [0, 255, 50], 2)						# Kişi sayısını görüntülemek için metin yerleştirin
    return img 												# Algılamalarla görüntüyü döndür
    #=================================================================#


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