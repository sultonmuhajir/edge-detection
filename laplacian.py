import cv2
import numpy as np
from matplotlib import pyplot as plt


#Memasukkan Gambar
img0 = cv2.imread('img/fruit.jpg',)


#Konfersi Greyscale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)


#Membuang Noise
img = cv2.GaussianBlur(gray,(3,3),0)


#Laplacian
laplacian = cv2.Laplacian(img,cv2.CV_64F)


#Menampilkan Hasil
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.show()