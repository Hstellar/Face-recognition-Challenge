
# coding: utf-8

# In[6]:


def extract_sift(img):    
    """
    Extract SIFT keypoints and descriptors from an image
    
    """

    import numpy as np
    import cv2
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img,None)    
    return kp,des


# In[8]:


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


# In[9]:


import glob
import cv2
import numpy as np
descriptors = []
labels = []
y = []
X=[]
temp=[]
name='E:/extra/ML/CV/Data'
train = glob.glob(name+"/*")
name2='E:/extra/ML/CV/TestSet'
test = glob.glob(name2+"/*")
n_sample=0
for myFile in train:
    if(myFile[len(name)+2]=='_'):
             label=myFile[len(name)+1]  
             print(label)
    if(myFile[len(name)+3]=='_'):
              label=myFile[len(name)+1:len(name)+3]
              print(label) 
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
for myFile in train:
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


# In[10]:



print("done")
svm = SVMClassifyer("rbf", 10, 0.00001, initial,labels)
print("done")


# In[ ]:





# In[42]:


accuracy = 0
c=0
X_test=[]
Y_test=[]
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


name2='E:/extra/ML/CV/TestSet'
test = glob.glob(name2+"/*")
total_imgs = len(test)
for img in test: 
    c=c+1
    if(img[len(name2)+2]=='_'):
             real_label=img[len(name2)+1]  
    if(img[len(name2)+3]=='_'):
              real_label=img[len(name2)+1:len(name2)+3]
    #print("Actual Label: "+real_label)
    X_test=np.append(X_test,real_label)
    im=cv2.imread(img,0)
    kp, des = extract_sift(im)
    des=np.array(des).flatten()
    #print(des)
    if maxx>des.shape[0]:
        des=np.append(des,np.zeros(maxx-des.shape[0]))
    des=des.reshape(1,des.shape[0])
    #print(des.shape)
    pred = svm.predict(des)
    #print("Predicted label: ") 
    #print(pred)
    Y_test=np.append(Y_test,pred)
    if real_label==pred:
         accuracy +=1
Y_test=np.array(Y_test)
X_test=np.array(X_test)
print("Actual label")
print(X_test)
print("predicted label")
print(Y_test)
#precision_recall_fscore_support(X_test, Y_test, average='weighted')   
x=recall_score(X_test, Y_test, average='weighted')
print("Recall= ",x)
y=precision_score(X_test, Y_test, average='weighted')
print("Precision = ",y)
accuracy = accuracy/total_imgs*100
print("Accuracy: ",accuracy)

