import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sn 
import warnings 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 

data = pd.read_csv('WineQT.csv')
print(data.head())
print(data.info())
print(data.isna().sum())#to check for null values
data = data.replace({'quality':{8:'good',7:'good',6:'average',5:'average',4:'bad',3:'bad'}})
print(data.head())