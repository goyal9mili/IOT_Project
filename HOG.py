import glob
import numpy as np
import cv2
import os
import numpy.random as nprnd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import math

def read_files(path):
    train_set = []
    test_set = []
    classes = []
    classes_counts = []
    dataset_classes = glob.glob(path + "/*")
    for folder in dataset_classes:
            path = folder.replace("\\", "/")
            if "/" in folder:
                class_name = folder.split("/")[-1]
            else:
                class_name = folder.split("\\")[-1]
            classes.append(class_name)
            train = glob.glob(path + "/train/*")
            test = glob.glob(path + "/test/*")
            train_set.append(train)
            test_set.append(test)
            classes_counts.append(0)        # why this line?
    return train_set, test_set, classes, classes_counts

train_set, test_set, class_list, classes_counts = read_files("dataset")
print(class_list)

def bilateral_filter (image):
    img = image
    bilateral = cv2.bilateralFilter(img, 15, 75, 75) 
    return bilateral

def canny_edge_detector (image):
    sigma = 0.33
    v = np.median(image)
    lower = int(max(0,(1.0 - sigma)*v))
    upper = int(min(255,(1.0 + sigma)*v))
    edged = cv2.Canny(image,lower,upper)
    return edged

def compute_hog(image):
    #hog = cv2.HOGDescriptor()
    winSize = (20,20)
    blockSize = (16,16)
    blockStride = (4,4)
    cellSize = (16,16)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True
    
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
    h = hog.compute(image)
    return h

def get_data_and_labels(img_set):
    y = []
    hog = []
    for class_number in range(len(img_set)):
        img_paths = img_set[class_number]
        #step = round(constants.STEP_PERCENTAGE * len(img_paths) / 100)
        for i in range(len(img_paths)):
            # if (step > 0) and (i % step == 0):
            #     percentage = (100 * i) / len(img_paths)
            #     print("Calculating global descriptors for image number {0} of {1}({2}%)".format(
            #         i, len(img_paths), percentage)
            #     )
            img = cv2.imread(img_paths[i])
            #print("des before vlad is", des)
            #vlad_vector = vlad(des, codebook)    
            #gabor_vector = gabor_filter(img)   
            #print(gabor_vector)
            bf = bilateral_filter(img)
            gray = cv2.cvtColor(bf, cv2.COLOR_BGR2GRAY)
            canny_edged = canny_edge_detector(gray)
            hog.append(compute_hog(canny_edged))    
            y.append(class_number)
            #hog.append(gabor_vector)
            
                    # x =d(class_number)
            # else:
            #     print("Img with None descriptor: {0}".format(img_paths[i]))
    y = np.float32(y)#[:, np.newaxis]
    hog = np.float32(hog)
    #x = np.float32(x)
    # hog = np.array(hog)
    # hog = np.float32(hog)
    return hog, y
        
x_train, y_train = get_data_and_labels(train_set)

n1, n2, n3 = x_train.shape
x_train = x_train.reshape(n1, n2*n3)

print(x_train)
print(y_train)

svclassifier = SVC(kernel = 'rbf', verbose=1)
svclassifier.fit(x_train,y_train)

x_test, y_test = get_data_and_labels(test_set)

n1, n2, n3 = x_test.shape
x_test = x_test.reshape(n1, n2*n3)

prediction_svm = svclassifier.predict(x_test)
print("SVM :",accuracy_score(y_test, prediction_svm))


