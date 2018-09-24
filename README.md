# Face-recognition-Challenge
Face recognition is implemented using SIFT features from facial images and then classifying by Support vector machine.
Step1. Read Dataset for training images.
Step2. Here vector named labels is made which contains all label name for images given in dataset.(Dataset used was of our academic class).
Step3. Calculate descriptor i.e feature matrix for all images, flatten them and store them in matrix named initial whose size is (no of sample)x(keypoints*128).
Step4. As we get different matrix size for different images so before providing it to Classifier the matrix should be in same size. I have calculated size for each matrix and store it in an array and then the maximum of it is taken as size of matrix
       Now append zeros to the matrix whose size is less than maximum size calculated and final matrix dimension should be (no of sample) X (maximum size).
Step5. Now provide it to SVM classifier and fit the model. Test images from dataset and predict result to calculate accuracy.
