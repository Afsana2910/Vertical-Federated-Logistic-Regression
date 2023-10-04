#This script is used to split the dataset into three partitions vertically
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('heart_disease.csv')


y = df["output"]
X = df.drop(["output"],axis=1)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

x1 = x_train.iloc[:, :4].values
x2 = x_train.iloc[:, 4:9].values
x3 = x_train.iloc[:, 9:].values

x1_test = x_test.iloc[:, :4].values
x2_test = x_test.iloc[:, 4:9].values
x3_test = x_test.iloc[:, 9:].values


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
