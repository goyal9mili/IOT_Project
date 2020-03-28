#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# In[3]:



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


# In[4]:


train_set, test_set, class_list, classes_counts = read_files("data")


# In[5]:


print(class_list)


# In[6]:


def resize(img, new_size, h, w):

    if h > w:
        new_h = 640
        new_w = (640 * w) / h
    else:
        new_h = (640 * h) / w
        new_w = 640
    img = cv2.resize(img, (new_w, new_h))
    return img


# In[7]:


def descriptors_from_class(class_img_paths, class_number, classes_counts):
    des = None
#     step = (20 * len(class_img_paths)) / 100
    for i in range(len(class_img_paths)):
        #print(i)
        img_path = class_img_paths[i]
        #print(img_path)
        img = cv2.imread(img_path)
        #print(img)
        resize_to = 640
        h, w, channels = img.shape
        if h > resize_to or w > resize_to:
            img = resize(img, resize_to, h, w)
        orb = cv2.ORB_create()
        kp, new_des = orb.detectAndCompute(img, None)
        # print("new_des created is", new_des)
        if new_des is not None:
            new_des = np.float32(new_des)
            #print("new_des dtype is", new_des.dtype)
            if des is None:
                des = np.array(new_des, dtype=np.float32)
            else:
                des = np.vstack((des, np.array(new_des)))
#         if i % step == 0:
#             percentage = (100 * i) / len(class_img_paths)
#             message = "Calculated {0} descriptors for image {1} of {2}({3}%) of class number {4} ...".format(
#                 des_name, i, len(class_img_paths), percentage, class_number
#             )
#             print(message)
#     message = "* Finished getting the descriptors for the class number {0}*".format(class_number)
    print("Number of descriptors in class: {0}".format(len(des)))
    classes_counts[class_number] = len(des)
    return des


# In[8]:


def orb_descriptors(class_list, classes_counts):
    des = None
    for i in range(len(class_list)):
        class_img_paths = class_list[i]
        #print("class_img_path is", class_img_paths)
        new_des = descriptors_from_class(class_img_paths, i, classes_counts)
        if des is None:
            des = new_des
        else:
            des = np.vstack((des, new_des))
            #print(" final des dtype",des.dtype)
#     message = "*****************************\n"\
#               "Finished getting all the descriptors\n"
#     print(message)
#     print("Total number of descriptors: {0}".format(len(des)))
#     if len(des) > 0:
#         print("Dimension of descriptors: {0}".format(len(des[0])))
#         print("First descriptor:\n{0}".format(des[0]))
    #print("................des type..", des.dtype)
    return des


# In[ ]:


des = orb_descriptors(train_set, classes_counts)
#print("................des type..", des.dtype)
#print(des)

# In[1]:


def generate_codebook(descriptors, k= 64):
        iterations = 10
        epsilon = 1.0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
        compactness, labels, centers = cv2.kmeans(descriptors, k, None, criteria, iterations, cv2.KMEANS_RANDOM_CENTERS)  # might have to add none here
        return centers

codebook = generate_codebook(des, 5)
#print(codebook)


def find_nn(point, neighborhood):
    min_dist = float('inf')
    nn = neighborhood[0]
    nn_idx = 0
    for i in range(len(neighborhood)):
        neighbor = neighborhood[i]
        dist = cv2.norm(point - neighbor)
        if dist < min_dist:
            min_dist = dist
            nn = neighbor
            nn_idx = i

    return nn, nn_idx

def vlad(descriptors, centers):
    #print("descriptor[0] is", descriptors[0])
    dimensions = len(descriptors[0])
    #print(dimensions)
    vlad_vector = np.zeros((len(centers), dimensions), dtype=np.float32)
    for descriptor in descriptors:
        nearest_center, center_idx = find_nn(descriptor, centers)
        for i in range(dimensions):
            vlad_vector[center_idx][i] += (descriptor[i] - nearest_center[i])
    # L2 Normalization
    vlad_vector = cv2.normalize(vlad_vector, None)
    vlad_vector = vlad_vector.flatten()
    return vlad_vector

def get_data_and_labels(img_set, codebook):
    y = []
    gf = []
    x = None
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
            orb = cv2.ORB_create()
            kp, des = orb.detectAndCompute(img, None)
            #print("des before vlad is", des)
            if des is not None:
                des = np.array(des, dtype=np.float32)
                vlad_vector = vlad(des, codebook)
                gabor_vector = gabor_filter(img)
                if x is None:
                    x = vlad_vector
                    gf = gabor_vector
                    y.append(class_number)
                else:
                    gf = np.vstack((gf, gabor_vector))
                    x = np.vstack((x, vlad_vector))
                    y.append(class_number)
            else:
                print("Img with None descriptor: {0}".format(img_paths[i]))
    y = np.float32(y)[:, np.newaxis]
    # print("vlad_vector shape is ...", vlad_vector.shape)
    # print("gabor_vector shape is ...",gabor_vector.sahpe)
    x = np.array(x)
    x = np.float32(x)
    gf = np.array(gf)
    gf = np.float32(gf)
    return x, gf, y

# def gabor_filter (image_set):
#     gabor_feature = []
#     labels = []
#     for class_number in range(len(image_set)):
#         img_paths = image_set[class_number]
#         #step = round(constants.STEP_PERCENTAGE * len(img_paths) / 100)
#         for i in range(len(img_paths)):
#             img = cv2.imread(img_paths[i])
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             img_pred = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
#             kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
#             kernel /= math.sqrt((kernel * kernel).sum())
#             g_f = cv2.filter2D(img_pred, -1, kernel)  
#             gabor_feature.append(g_f)
#             labels.append(class_number)
#     return gabor_feature, labels


def gabor_filter (image):
    img = image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pred = cv2.resize(img, (160, 1), interpolation=cv2.INTER_AREA)
    kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
    kernel /= math.sqrt((kernel * kernel).sum())
    g_f = cv2.filter2D(img_pred, -1, kernel)
    return g_f 

# gabor_feature_train, labels_train_gabor = gabor_filter(train_set)
vlad_feature_train, gabor_feature_train, y_train = get_data_and_labels(train_set, codebook)

print("vlad_features are.............", vlad_feature_train)
print("vlad features size: ", vlad_feature_train.shape)
print("gabor_feature_train...........", gabor_feature_train)
print("gabor faetures size : ",gabor_feature_train.shape)

# vlad_feature_train.resize(gabor_feature_train)
A = (vlad_feature_train, gabor_feature_train)
x_train = np.vstack(A)
B = (y_train, y_train)
y_train = np.vstack(B)
# d_train = np.column_stack((np.array(gabor_feature_train), np.array(vlad_feature_train)))
# x_train = np.array(d_train)

# print(x_train)
#print(y_train)

# svm = cv2.ml.SVM_create()
# svm_params = dict(kernel_type=cv2.ml.SVM_RBF, svm_type=cv2.ml.SVM_C_SVC, C=1)
# svm.train_auto(x_train, y_train, svm_params)

svclassifier = SVC(kernel = 'rbf', verbose=1)
svclassifier.fit(x_train,y_train)

rf_clf = RandomForestClassifier(n_estimators = 100, random_state = 4, n_jobs=2, verbose = True)
rf_clf.fit(x_train, y_train)

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

vlad_feature_test, gabor_feature_test, y_test = get_data_and_labels(test_set, codebook)

# vlad_feature_test.resize(gabor_feature_test)
C = (vlad_feature_test, gabor_feature_test)
x_test = np.vstack(C)
D = (y_test, y_test)
y_test = np.vstack(D)
# gabor_feature_test, labels_test_gabor = gabor_filter(test_set)

# d_test = np.column_stack((np.array(gabor_feature_test), np.array(vlad_feature_test)))
# x_test = np.array(d_test)

# result = svm.predict_all(x_test)
# mask = result == y_test
# correct = np.count_nonzero(mask)
# accuracy = (correct * 100.0 / result.size)
# print("accuracy is", accuracy)

prediction_svm = svclassifier.predict(x_test)
print("SVM :",accuracy_score(y_test, prediction_svm))

predictions_rf = rf_clf.predict(x_test)
print("RF :",accuracy_score(y_test, predictions_rf))

predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print("Logistic :",score)


# In[2]:

 

# In[ ]:





# In[ ]:




