# -*- coding: utf-8 -*-
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import seaborn as sns #je vais l'uitiliser pour sns.boxplot
from  sklearn.metrics import accuracy_score , confusion_matrix, log_loss
from sklearn.preprocessing import OneHotEncoder, LabelEncoder 
from sklearn.model_selection import cross_val_score , train_test_split
from sklearn.model_selection import StratifiedKFold # pour balancer dataset 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV  #for search the best tuning
import time 
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
from sklearn.metrics import plot_confusion_matrix
data = pd.read_csv('C:\\Users\\hp\\Desktop\\result\\dta.csv')
print('data infois:',data.info())
data = data.drop('id', axis= 1)
# split data into X and y
obs= data.shape[0]
end= obs-24
X = data.iloc[:end,:]# commence du jour j à (jour_final-1)
x = pd.DataFrame(data= X, columns= data.columns)
corr = data.corr()
y = data.iloc[:,2]


for i  in range(len(y)) :
    
    if y.iloc[i] > 0:# il pleut.
        y[i]= 1
    else:
        y[i]= 0  # il ne pleut pas.
y = data.iloc[24:,2]# commence du jour (j+1) à jour_final      

y = pd.DataFrame(data= {'target': y}).astype(int)

print(y)

# the scaler object (model)
normalization = MinMaxScaler()
# fit and transform the data
X = normalization.fit_transform(X)

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = 
                  train_test_split(X, y,test_size=test_size,
                                   random_state=seed)        

# construire le classifier model

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
****
classifier  = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train, y_train)
classifier.predict(X_test)
y_pred= round(classifier.score(X,y), 0)

y_pred = svclassifier.predict(X_test)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#y_pred = pd.DataFrame(data= {'predict': [y_pred]}).astype(int)


ind = np.zeros(len(y_pred))
indd=0
for i in range(len(y_pred)):
    ind[i]= int(indd) 
    indd= indd+1
    indd= int(indd)
    

y_pred = pd.DataFrame(y_pred,index=ind).round(0)

print('real is :',y_test )
print( 'shape of real', y_test.shape)
print('predict is :',y_pred )
print( 'shape of predict', y_pred.shape)

print('type of real is :', type(y_test))
print('type of predict is :', type(y_pred))

print (classification_report(y_test, y_pred))


labels= ['il pleut','il ne pleut pas']

cf_matrix = confusion_matrix(y_test,y_pred ,labels)

print(cf_matrix)
'''
print(cm)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
'''
#plot_confusion_matrix(classifier, X_test, y_test)  # doctest: +SKIP

import seaborn as sns
group_names = ['True Negatif','False Positif','False Negatif',
               'True Positif']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

# classifier model:  SVC kernel


svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#y_pred = pd.DataFrame(data= {'predict': [y_pred]}).astype(int)


ind = np.zeros(len(y_pred))
indd=0
for i in range(len(y_pred)):
    ind[i]= int(indd) 
    indd= indd+1
    indd= int(indd)
    

y_pred = pd.DataFrame(y_pred,index=ind).round(0)

print('real is :',y_test )
print( 'shape of real', y_test.shape)
print('predict is :',y_pred )
print( 'shape of predict', y_pred.shape)

print('type of real is :', type(y_test))
print('type of predict is :', type(y_pred))

print (classification_report(y_test, y_pred))


labels= ['il pleut','il ne pleut pas']

cf_matrix = confusion_matrix(y_test,y_pred ,labels)

print(cf_matrix)
'''
print(cm)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
'''
#plot_confusion_matrix(classifier, X_test, y_test)  # doctest: +SKIP

import seaborn as sns
group_names = ['True Negatif','False Positif','False Negatif',
               'True Positif']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')







