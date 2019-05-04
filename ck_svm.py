
from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio # 用来读取matlab数据
import csv
from sklearn.cross_validation import train_test_split
#from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix




_ck_landmarks = '/home/yixin/dissertation/Deep-Facial-Expression-Recognition/data/CK+_landmarks_7exp.csv'
_ck_10folds = '/home/yixin/dissertation/Deep-Facial-Expression-Recognition/data/CK+_10folds_7exp.pkl'

_phog_feature_ck = '/home/yixin/dissertation/PHOG/_phog_feature_7exp_ck+.pkl'
_lpq_feature_ck = '/home/yixin/dissertation/LPQpy/_lpq_feature_7exp_ck+.pkl'





# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
# Download the training data
file = open(_phog_feature_ck, "rb")
data = pickle.load(file)
feature = data['phog_feature']

df = pd.read_csv(_ck_landmarks, delimiter=",")
labels = df['img_label'].values

def get_targets():
    unique_class_id = np.unique(labels)
    unique_class_id = sorted(unique_class_id)
    le = preprocessing.LabelEncoder()
    le.fit(unique_class_id)
    targets = le.transform(labels).reshape(-1, 1)
    num_class = len(unique_class_id)
    return targets.astype(np.float32), num_class


def getRecognitionRate(testPre, testClass):
    testNum = len(testPre)
    rightNum = 0
    for i in range(0, testNum):
        if testClass[i] == testPre[i]:
            rightNum += 1
    return float(rightNum) / float(testNum)

targets, num_class = get_targets()

target_names = {'0': 'AN', '1': 'DI', '2': 'FE', '3': 'HA', '4': 'NE', '5': 'SA', '6': 'SU'}

x_train, x_test, y_train, y_test = train_test_split(feature, targets, test_size=0.1, random_state=42)


n_components = 98
pca = PCA(n_components=n_components, whiten=True).fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

clf = SVC(C=10, gamma=0.01, kernel='rbf', class_weight='balanced', max_iter=1000)
clf = clf.fit(x_train_pca, y_train.ravel())

y_pred = clf.predict(x_test_pca)
y_pred = np.array(y_pred).reshape(len(y_pred),1)


# print(classification_report(y_test, y_pred, target_names=target_names_test))
cmatrix = confusion_matrix(y_test, y_pred)
y_true_sum = cmatrix.sum(axis=1)
y_true_sum = y_true_sum.clip(min=1e-6)  # prevent divide zero
cmatrix_normalized = cmatrix / y_true_sum
mean_acc = cmatrix_normalized.diagonal().mean()
RR = getRecognitionRate(y_pred, y_test)
message = "CK+ acc.:{} RR:{}".format(mean_acc, RR)
print(message)
#RecognitionRate








