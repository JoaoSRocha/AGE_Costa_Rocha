#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:47:00 2018

@author: togepi
"""

import AGEmodule as AGE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score

feats, labels = AGE.load_features_and_labels()

#%% train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(feats, labels)

#%%
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

feature_processing = RobustScaler()
feature_selection = SelectPercentile(mutual_info_classif, percentile = 45)
svm_clf = SVC(max_iter = -1, probability = True, kernel='rbf', C = 10, gamma = 0.001)

pipe = Pipeline([('scale', feature_processing), 
                 ('selection', feature_selection),
                 ('classifier', svm_clf)])

pipe.fit(Xtrain, ytrain)

cr = classification_report(ytest, pipe.predict(Xtest))

#%% ROC curve
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

classes = np.arange(1,27,1)
n_classes = len(classes)
ytest_bin = label_binarize(ytest, classes)
Xtest_dec = pipe.predict_proba(Xtest)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(ytest_bin[:, i], Xtest_dec[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(ytest_bin.ravel(), Xtest_dec.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#%%
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
    plt.figure()
    lw = 2
    plt.plot(fpr[USER], tpr[USER], color='darkorange',
             lw=lw, label='ROC curve (area = %0.8f)' % roc_auc[USER])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#%%
import matplotlib.pyplot as plt

def plot_roc_user(USER):
    plt.title('Receiver operating characteristic ('+str(USER)+')')
    plt.legend(loc="lower right")
    plt.show()

#%%
plot_roc_user('micro')
