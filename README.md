# Face-recognition-Challenge
Face recognition is implemented using SIFT features from facial images and then classifying by Support vector machine.<br />
Step1. Read Dataset for training images.<br />
Step2. Here vector named labels is made which contains all label name for images given in dataset.(Dataset used was of our academic class).<br />
Step3. Calculate descriptor i.e feature matrix for all images, flatten them and store them in matrix named initial whose size is (no of sample)x(keypoints*128).<br />
Step4. As we get different matrix size for different images so before providing it to Classifier the matrix should be in same size. I have calculated size for each matrix and store it in an array and then the maximum of it is taken as size of matrix<br />
       Now append zeros to the matrix whose size is less than maximum size calculated and final matrix dimension should be (no of sample) X (maximum size).<br />
Step5. Now provide it to SVM classifier and fit the model. Test images from dataset and predict result to calculate accuracy.<br />
## Face recognition using Singular value decomposition
Dataset used was of yalefaces. All the faces are stacked in matrix and normalization is done. SVD is calculated of that matrix and U matrix obtained is multiplied with faces(f x U). This is done for training dataset. For test image(t) U matrix is obtained from SVD and multiplied with test image i.e t x U. Now distance or norm is calculated from test image to train images. The image with which least norm is obtained is the correct image.

Code was built in python in spyder3.6. Here test image taken was one image from dataset.
