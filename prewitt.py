import cv2
import numpy as np
from matplotlib import pyplot as plt


#Memasukkan Gambar
img0 = cv2.imread('img/fruit.jpg',)


#Konfersi Greyscale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)


#Membuang Noise
img = cv2.GaussianBlur(gray,(3,3),0)


#Prewit (deteksi tepi sesuai koordinat)
prex = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
prey = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittx = cv2.filter2D(img, -1, prex)
prewitty = cv2.filter2D(img, -1, prey)
prewitt = prewittx + prewitty


#Menampilkan Hasil
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(prewitt,cmap = 'gray')
plt.title('Prewitt'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(prewittx,cmap = 'gray')
plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(prewitty,cmap = 'gray')
plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])

plt.show()
