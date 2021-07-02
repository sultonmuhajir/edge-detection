import cv2
import numpy as np
from matplotlib import pyplot as plt


#Memasukkan Gambar
img0 = cv2.imread('img/fruit.jpg')


#Konfersi Greyscale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)


#Membuang Noise
img = cv2.GaussianBlur(gray,(3,3),0)


#Sobel (deteksi tepi sesuai koordinat)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
sobel = sobelx + sobely


#Menampilkan Hasil
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(sobel,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
