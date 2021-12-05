from os import closerange
import sklearn
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from random import sample
import random
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
df = pd.read_csv(r'/home/afsana/FL/Codes/dataset/heart_disease_health_indicators.csv')
features = df.columns.tolist()
#random.shuffle(features)
#df = df[features]
##print(df.columns)
X = df.drop(["HeartDiseaseorAttack"],axis=1)
y = df["HeartDiseaseorAttack"]
print(X.columns)
#print(X.shape,y.shape)
rus = RandomUnderSampler(sampling_strategy='majority',random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)
#print(sorted(Counter(y_resampled).items()))
#print(X_resampled.shape,y_resampled.shape)
s = 30000
X = X_resampled.values
y = y_resampled.values


scaler = StandardScaler()
X = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
x_train = x_train[:3000,:]
y_train = y_train[:3000]
x_test = x_test[3000:4500,:]
y_test = y_test[3000:4500]
#x_train , x_test= X[:s,:], X[s:,:]
x1,x2,x3 = x_train[:,:7], x_train[:,7:14], x_train[:,14:]
x1_test,x2_test,x3_test= x_test[:,:7],x_test[:,7:14],x_test[:,14:]

#y_train = y[:s]
#y_test = y[s:]
#print(x_train.shape)
#print(x_test.shape)
def get_data():
    return x1,x2,x3
    
def get_labels():
    return y_train

def get_testdata():
    return x1_test,x2_test,x3_test

def get_testlabels():
    return y_test

def get_dataset():
    return x_train, x_test

