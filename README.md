# Kurulum

## Repo'nun klonlanması
 - git clone https://github.com/aygen-ergen/Darknet-Yolo-ile-Sosyal-Mesafe-Uygulamas-.git
  
## Makefile dosyasının düzenlenmesi
  - %cd /content/darknet
  - sed -i 's/OPENCV=0/OPENCV=1/' Makefile
  - sed -i 's/GPU=0/GPU=1/' Makefile
  - sed -i 's/CUDNN=0/CUDNN=1/' Makefile
   Eğer kendi ekran kartınızı kullanmak istemezseniz, yalnızca OpenCV satırını çalıştırın.
   
## Cuda'nın doğrulanması
  - nvcc --version
  
## Darknet repo'sunun derlenmesi
  - make
  
## Sosyal mesafe uygulamasının çalıştırılması
  - python3 /darknet/App1_Sosyal_Mesafe.py
  
## İnsan yoğunluğu uygulamasının çalıştırılması
  - python3 /darknet/insanYogunlugu.py
