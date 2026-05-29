import pandas as pd 
import joblib 
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
from sklearn.metrics import classification_report , confusion_matrix

data = pd.read_csv('WineQT.csv')
print(data.head())
print(data.info())
print(data.isna().sum())#to check for null values
data = data.replace({'quality':{8:'good',7:'good',6:'average',5:'average',4:'bad',3:'bad'}})
print(data.head())

x = data.drop(columns = 'quality')
y = data['quality']
scaler = MinMaxScaler(feature_range=(0,1))
x_scaled = scaler.fit_transform(x)
x_train , x_test , y_train , y_test = train_test_split(x_scaled , y , test_size = .2 , random_state = 42 )

models= {
    "logisitic regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}
result = {}
for model_name , model in models.items():
    print(f"Training {model_name}....🤖")
    model.fit(x_train , y_train)
    y_pred = model.predict(x_test)
    acc = metrics.accuracy_score(y_test , y_pred)
    result[model_name] = acc 
    print(f"{model_name} Accuracy:{acc:.4f}")

sorted_result = sorted(result.items(), key=lambda x: x[1] , reverse=True)
print("\nModel Performance:")
for model_name , acc in sorted_result:
    print(f"{model_name}: {acc:.4f}")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv = 5 ,  n_jobs = -1 , verbose = 1)
grid_search.fit(x_train , y_train)
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
cv_scores = cross_val_score(best_model, x_scaled, y, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f}")

y_pred = best_model.predict(x_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

#save best model 
joblib.dump(best_model , 'model.pkl')
joblib.dump(scaler , 'scaler.pkl')
print("Model and scaler saved successfully.")