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

train_set, test_set, class_list, classes_counts = read_files("data")
print(class_list)

def gabor_filter (image):
    img = image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_pred = cv2.resize(img, (160,160), interpolation=cv2.INTER_AREA)
    kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
    kernel /= math.sqrt((kernel * kernel).sum())
    g_f = cv2.filter2D(img, -1, kernel)
    #print(g_f)
    return g_f

def get_data_and_labels(img_set):
    y = []
    gf = []
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
            gf.append(gabor_filter(img))    
            y.append(class_number)
            #gf.append(gabor_vector)
            
                    # x =d(class_number)
            # else:
            #     print("Img with None descriptor: {0}".format(img_paths[i]))
    y = np.float32(y)#[:, np.newaxis]
    gf = np.float32(gf)
    #x = np.float32(x)
    # gf = np.array(gf)
    # gf = np.float32(gf)
    return gf, y
        
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

# predictions_rf = rf_clf.predict(x_test)
# print("RF :",accuracy_score(y_test, predictions_rf))

# predictions = logisticRegr.predict(x_test)
# score = logisticRegr.score(x_test, y_test)
# print("Logistic :",score)

