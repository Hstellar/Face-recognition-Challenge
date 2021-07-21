# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 23:48:33 2018

@author: HENA
"""
import glob
import cv2
import numpy as np
from numpy import linalg 
from PIL import Image
h=50
w=50

n_sample=0
path=r"E:\music\yaleface_jpg\\"
files = glob.glob(path+"/*")
no_of_images=165
X = np.empty(shape=(h*w,no_of_images), dtype='float64')
for myFile in files:
    img = cv2.imread (myFile,0)
    image = cv2.resize(img, (h, w), interpolation = cv2.INTER_AREA) 
    gray_vector = np.array(image, dtype='float64').flatten()
    X[:, n_sample] = gray_vector[:] 
    n_sample=n_sample+1


mean = np.sum(X, axis=1) /no_of_images
#subtracting mean matrix from X matrix

for index in range(no_of_images):
    X[:,index] = X[:,index]- mean[:]

#singular value decomposition    
u, s, v = np.linalg.svd(X, full_matrices=False)
fu = np.empty(shape = (u.shape[0]*u.shape[1], u.shape[1]),  dtype=np.int8)

# temporary array
temp = np.empty(shape = (u.shape[0], u.shape[1]),  dtype=np.int8)

#Matrix that stores dot product of U and train images X
for i in range(X.shape[1]): 
    for k in range(u.shape[1]):    
        temp[:,k] = X[:,i] * u[:,k]    
    f1 = np.array(temp, dtype='int8').flatten()   
    fu[:, i] = f1[:]
    #print(fu[:, i])
#print(fu.shape)
    
test_face = np.empty(shape = (u.shape[0], u.shape[1]),  dtype=np.int8)
path_t=r"E:\music\test\\"  #path for test images
files_t = glob.glob(path_t+"/*")
i=0
for myFile in files_t:
    print("input face: ",myFile)
    img = cv2.imread (myFile,0)
    image = cv2.resize(img, (h, w), interpolation = cv2.INTER_AREA) 
    test = np.array(image, dtype='float64').flatten() 
    test =test- mean
    test_face[:,i]=test[:]
    i=i+1


#Matrix that stores dot product of U and test image
temp = np.empty(shape = (u.shape[0], u.shape[1]),  dtype=np.int8)

for col in range(u.shape[1]):    
    temp[:,col] = test_face[:,0] * u[:,col]

temp1 = np.array(temp, dtype='int8').flatten()

#distance matrix 
d = np.empty(shape = (u.shape[0]*u.shape[1], u.shape[1]))

for col in range(u.shape[1]):
    d[:,col] = fu[:,col] - temp1[:]
mag = np.empty(shape=(u.shape[1],))

for c in range(d.shape[1]):    
    mag[c] = np.linalg.norm(d[:,c])

index=np.argmin(mag)
print(index)
i=0
for myFile in files:
    if i==index:
        print("recognized image: ",myFile)
        break
    else:
        i=i+1
