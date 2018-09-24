
# coding: utf-8

# In[1]:


def extract_sift(img):    
    """
    Extract SIFT keypoints and descriptors from an image
    
    """

    import numpy as np
    import cv2
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img,None)    
    return kp,des


# In[2]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

"""
   Input: SVM parameters (kernel, C, gamma), data and labels coresponding to the data
   output: Trained SVM classifier ready to be used for prediction
"""
def SVMClassifyer(k, c, g, data, labels):
    clf = SVC(kernel =k, C =c, gamma = g)
    clf.fit(data, labels)
    return clf

"""
   Input: Classifier, range of values for the parameters,  data and labels coresponding to the data
   output: Best parameters with the corresponding score
"""
def gridSearch(clf,C_range, gamma_range,crossVal, data, labels):
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(clf, param_grid=param_grid, cv=crossVal)
    grid.fit(data, labels)
    return grid.best_params_, grid.best_score_


# In[31]:


import glob
import cv2
import numpy as np
descriptors = []
labels = []
y = []
X=[]
temp=[]
name='E:/extra/ML/CV/DataSet'
train = glob.glob(name+"/*")
name2='E:/extra/ML/CV/TestSet'
test = glob.glob(name2+"/*")
n_sample=0
for myFile in train:
    #print(myFile)
    if(myFile[len(name)+2]=='_'):
             label=myFile[len(name)+1]  
             #print(label)
    if(myFile[len(name)+3]=='_'):
              label=myFile[len(name)+1:len(name)+3]
              #print(label) 
    im=cv2.imread(myFile,0)
    n_sample=n_sample+1
    kp, des = extract_sift(im)
    des=np.array(des)
    temp.append(des.shape[0]*des.shape[1]) 
    labels.append(label)
labels=np.array(labels)        
temp=np.array(temp)
maxx=np.amax(temp)
initial=np.zeros((n_sample,maxx))
i=0
for myFile in test:
    im=cv2.imread(myFile,0)
    kp, des = extract_sift(im)
    des=np.array(des).flatten()
    #print(des.shape)
    if maxx>des.shape[0]:
        des=np.append(des,np.zeros(maxx-des.shape[0]))
    des=des.reshape(1,des.shape[0])
    initial[i,:]=des[:,:]
    i=i+1
print(initial.shape)
print(labels.shape)


# In[32]:


print("done")
svm = SVMClassifyer("linear", 10, 0.00001, initial,labels)
print("done")


# In[29]:


accuracy = 0
c=0
total_imgs = len(test)
for img in test: 
    c=c+1
    if(img[len(name2)+2]=='_'):
             real_label=img[len(name2)+1]  
    if(img[len(name2)+3]=='_'):
              real_label=img[len(name2)+1:len(name2)+3]
    print(img)
    im=cv2.imread(img,0)
    kp, des = extract_sift(im)
    des=np.array(des).flatten()
    print(des)
    if maxx>des.shape[0]:
        des=np.append(des,np.zeros(maxx-des.shape[0]))
    des=des.reshape(1,des.shape[0])
    print(des.shape)
    pred = svm.predict(des)
    print(pred)  
    if real_label==pred:
         accuracy +=1
accuracy = accuracy/total_imgs*100
print(accuracy)
print(c)

