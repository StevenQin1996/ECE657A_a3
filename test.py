import numpy as np
import pandas as pd
import random
import seaborn as sns
import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

scaler = StandardScaler()
train_set= pd.read_csv("train.csv",sep = ',',index_col = 0)
test_set= pd.read_csv("testX.csv",sep = ',',index_col = 0)
print(test_set)
#copy a new train set to analysis
temp_train = train_set[:]
temp_test = test_set[:]
del temp_train['Label']
X = temp_train.values
y = temp_test.values
#use normalization
X_std = scaler.fit_transform(X)
y_std = scaler.fit_transform(y)
pca = PCA(n_components=100,random_state = 42)
start_train_pca = time.time_ns()
train = pca.fit_transform(X_std)
end_train_pca = time.time_ns()
time_train_pca = (end_train_pca-start_train_pca)/1000000000
print("PCA train time:{}s".format(time_train_pca))
start_test_pca = time.time_ns()
test = pca.transform(y_std)
end_test_pca = time.time_ns()
time_test_pca = (end_test_pca-start_test_pca)/1000000000
print("PCA test time:{}s".format(time_test_pca))


k = [1,5,10,15,20,25,30,35]
for i in range(len(k)):
    neigh = KNeighborsClassifier(n_neighbors=k[i])
    start_train_knn = time.time_ns()
    neigh.fit(train, train_set['Label'].values)
    end_train_knn = time.time_ns()
    time_train_knn = (end_train_knn-start_train_knn)/1000000000
    print("knn train time:{}s".format(time_train_knn))
    scores = neigh.score(train, train_set['Label'].values)
    print("rt score:{}".format(scores))
    start_test_knn = time.time_ns()
    result_knn = neigh.predict(test)
    end_test_knn = time.time_ns()
    time_test_knn = (end_test_knn-start_test_knn)/1000000000
    print("knn test time:{}s".format(time_test_knn))
    dataframe = pd.DataFrame({'Label':result_knn})
    dataframe.to_csv("result{num}.csv".format(num = i),index=False,sep=',')



tempC = [0.1,0.5,1,2,5,10,20,50]
for i in range(len(tempC)):
    clf_svc = SVC(C= tempC[i],random_state=42)
    start_train_svc = time.time_ns()
    clf_svc.fit(train, train_set['Label'].values)
    end_train_svc = time.time_ns()
    time_train_svc = (end_train_svc-start_train_svc)/1000000000
    print("svc train time:{}s".format(time_train_svc))
    scores = cross_val_score(clf_svc,train, train_set['Label'].values,cv=10)
    print("svc score:{}".format(scores.mean()))
    start_test_svc = time.time_ns()
    result_svc = clf_svc.predict(test)
    end_test_svc = time.time_ns()
    time_test_svc = (end_test_svc-start_test_svc)/1000000000
    print("svc test time:{}s".format(time_test_svc))
    dataframe = pd.DataFrame({'Label':result_svc})
    dataframe.to_csv("result_svc{numi}.csv".format(numi = i),index=False,sep=',')

