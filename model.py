import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClasiifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd. read_csv ("/content/crop_recommendation.csv")
data.head (5)
data.shape
data.isnull().sum()
data.info()
data.describe()
data.nunique()
data['label'].value_counts()

X=data.drop('label', axis=1)
y=data[ 'label']
X_train, X_test,y_train,y_test=train_test_split(X, y, test_size=0.30, shuffle=True, random_state=0)
model= RandomForestClasiifier()
model.fit (X_train, y_train)
y_pred = model.predict (X_test)
accuracy=accuracy_score(y_pred, y_test)

 
cm = confusion_matrix(y_test, y_pred)
plt. figure (figsize=(15,15))
sns.heatmap (cm, annot=True, fmt=".0f", linewidths=.5, square= True, cmap = 'Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix - score: '+str(accuracy_score (y_test,y_pred))
plt.title(all_sample_title, size = 15);
plt.show()

 
print (classification_report (y_test,y_pred))
