import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os

def DataReading():
    filepath = os.path.dirname(__file__)
    file_train = pd.read_csv(filepath + '/criteo/train.tiny.csv')
    file_test = pd.read_csv(filepath + '/criteo/test.tiny.csv')
    file = pd.concat([file_train, file_test], axis=0)
    return file, file_train.shape[0], file_test.shape[0]

def DataCleaning(df):
    df = df.drop('Id', axis = 1)
    for col in list(df.columns[df.isnull().sum() > 0]):
        if df[col].dtype == 'int64':
            val = int(df[col].mean())
            df[col].fillna(val, inplace = True)
        elif df[col].dtype == 'float64':
            val = df[col].mean()
            df[col].fillna(val, inplace = True)
        else:
            val = df[col].value_counts().head(1).index[0]
            df[col].fillna(val, inplace = True)
    return df

def DataConverting(df):
    df_sparse = pd.get_dummies(df.select_dtypes(include=['object']))
    df_dense = df.drop([i for i in df.columns if i[0] == 'C'], axis=1)
    df_final = pd.concat([df_dense, df_sparse], axis=1)
    return df_final

def DataPreProcessing(df):
    Y = df.iloc[:, 0]
    X = df.iloc[:, 1:-1]
    X_scaled = preprocessing.scale(X)
    return X_scaled, Y

def DataSpliting(X, Y, train_ratio, test_ratio):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=train_ratio)
    Y_train = np.array(Y_train).reshape(-1, 1)
    Y_test = np.array(Y_test).reshape(-1, 1)
    Y_val = np.array(Y_val).reshape(-1, 1)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def DataLoading(train_ratio, test_ratio):
    f, s1, s2 = DataReading()
    df = DataCleaning(f)
    df = DataConverting(df)
    X, Y = DataPreProcessing(df)
    X_train, Y_train, X_test, Y_test, X_val, Y_val = DataSpliting(X, Y, train_ratio, test_ratio)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

if __name__ == '__main__':
    pass