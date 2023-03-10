# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 11:20:51 2022

multi classification using ANN, RandomForest and Support Vector Machine and Autokeras

@author: PRO
"""
# Classification Using ANN
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("C:/Users/PRO/Downloads/Untitled spreadsheet - Sheet1.csv")

print(df.describe().T)# To print the data information

print(df.isnull().sum()) #how many row do we have

#df = df.dropna() # to drop the null value

#Rename Dataset to Label to make it easy to understand
#df = df.rename(columns={'Diagnosis':'Label'}) # diagnosis from its orignal label to word label
#print(df.dtypes)

#Understand the data 
sns.countplot(x="Label", data=df) #apple,orange, banana

####### Replace categorical values with numbers########
print("Distribution of data: ", df['Label'].value_counts())

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values
print("Labels before encoding are: ", np.unique(y)) #it will print the value of the label

# Encoding categorical data from text (B and M) to integers (0 and 1)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y) # it will turn to numbercal value
print("Labels after encoding are: ", np.unique(Y))

from  tensorflow.keras.utils import to_categorical 
Y=(to_categorical(Y))
print(Y)

#Define x and normalize / scale values

#Define the independent variables. Drop label and ID, and normalize other data
X_df = df.drop(labels = ["Label"], axis=1) # change the label and ID
print(X_df.describe().T) #Needs scaling

#Scale / normalize the values to bring them to similar range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_df)
X = scaler.transform(X_df)
print(X)  #Scaled values

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print("Shape of training data is: ", X_train.shape)
print("Shape of testing data is: ", X_test.shape)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(384, input_dim=18, activation='relu')) #please take note the input_dim, this will be the number of parameters (the x column)
model.add(Dropout(0.3))
model.add(Dense(18,activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(3)) 
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

#Fit with no early stopping or other callbacks
history = model.fit(X_train, y_train, verbose=1, epochs=200, batch_size=64,
                    validation_data=(X_test, y_test))

#plot the train_test_split()ining and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']  #Use accuracy if acc doesn't work
val_acc = history.history['val_accuracy']  #Use val_accuracy if acc doesn't work
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


from sklearn.metrics import confusion_matrix
 
y_predicted = model.predict(X_test)
mat = confusion_matrix(y_test.argmax(axis=1), y_predicted.argmax(axis=1))
 
sns.heatmap(mat,annot=True)


#sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
 #           xticklabels=X.target_names,
 #           yticklabels=X.target_names)
 
#plt.xlabel('Predicted label')
#plt.ylabel('Actual label')

from sklearn.metrics import confusion_matrix,classification_report

y,levels = pd.factorize(df['Label'])

#mat = confusion_matrix(y_test, y_predicted)
#rndf=sns.heatmap(mat,annot=True)
#rndf.set_xlabel("Predicted label", fontsize = 20)
#rndf.set_ylabel("Actual label", fontsize = 20)

#cf_matrix = pd.crosstab(levels[y_test.argmax(axis=1)],levels[y_predicted.argmax(axis=1)])
fig, ax = plt.subplots(figsize=(5,5),)
plt.title("Artificial Neural Network Confusion Matrics",fontsize=20)

Ann=sns.heatmap(mat, linewidths=1, annot=True, ax=ax, fmt='g')
Ann.set_xlabel("Predicted label", fontsize = 10)
Ann.set_ylabel("Actual label", fontsize = 10)

#
print(classification_report(y_test.argmax(axis=1),y_predicted.argmax(axis=1),target_names=levels))



#######################################################################################


"classification using SVM check out for more datail:"
"https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


df = pd.read_csv("C:/Users/PRO/Downloads/Untitled spreadsheet - Sheet1.csv")

print(df.describe().T)# To print the data information

print(df.isnull().sum()) #how many row do we have

#df = df.dropna() # to drop the null value

#Rename Dataset to Label to make it easy to understand
#df = df.rename(columns={'Diagnosis':'Label'}) # diagnosis from its orignal label to word label
#print(df.dtypes)

#Understand the data 
sns.countplot(x="Label", data=df) #apple,orange, banana

####### Replace categorical values with numbers########
print("Distribution of data: ", df['Label'].value_counts())

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values
print("Labels before encoding are: ", np.unique(y)) #it will print the value of the label

# Encoding categorical data from text (apple and orange and mango) to integers (0, 1 and 2)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y) # it will turn to numbercal value
print("Labels after encoding are: ", np.unique(Y))


#Define x and normalize / scale values

#Define the independent variables. Drop label and ID, and normalize other data
X_df = df.drop(labels = ["Label"], axis=1) # change the label and ID
print(X_df.describe().T) #Needs scaling

#Scale / normalize the values to bring them to similar range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_df)
X = scaler.transform(X_df)
print(X)  #Scaled values

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print("Shape of training data is: ", X_train.shape)


linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)


#stepsize in the mesh, it alters the accuracy of the plotprint
#to better understand it, just play with the value, change it and print it
h = .01
#create the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
# create the title that will be shown on the plot
titles = ['Linear kernel','RBF kernel','Polynomial kernel','Sigmoid kernel']

"""for i, clf in enumerate((linear, rbf, poly, sig)):
    #defines how many plots: 2 rows, 2columns=> leading to 4 plots
    plt.subplot(2, 2, i + 1) #i+1 is the index
    #space between plots
    plt.subplots_adjust(wspace=0.4, hspace=0.4) 
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y,cmap=plt.cm.PuBuGn,edgecolors='grey')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
    plt.show()
"""
# confusion metrics for SVM linear
from sklearn.metrics import confusion_matrix,classification_report
linear_pred = linear.predict(X_test)
y,levels = pd.factorize(df['Label'])

#mat = confusion_matrix(y_test, y_predicted)
#rndf=sns.heatmap(mat,annot=True)
#rndf.set_xlabel("Predicted label", fontsize = 20)
#rndf.set_ylabel("Actual label", fontsize = 20)

cf_matrix = pd.crosstab(levels[y_test],levels[linear_pred])
fig, ax = plt.subplots(figsize=(5,5),)
plt.title("MSV for linear Model Confusion Matrics",fontsize=20)

MSV_linear=sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')
MSV_linear.set_xlabel("Predicted label", fontsize = 10)
MSV_linear.set_ylabel("Actual label", fontsize = 10)

print(classification_report(y_test,linear_pred,target_names=levels))

# confusion metrics for SVM polynomial
from sklearn.metrics import confusion_matrix,classification_report
poly_pred = poly.predict(X_test)
y,levels = pd.factorize(df['Label'])

#mat = confusion_matrix(y_test, y_predicted)
#rndf=sns.heatmap(mat,annot=True)
#rndf.set_xlabel("Predicted label", fontsize = 20)
#rndf.set_ylabel("Actual label", fontsize = 20)

cf_matrix = pd.crosstab(levels[y_test],levels[poly_pred])
fig, ax = plt.subplots(figsize=(5,5),)
plt.title("MSV for Polynomial Model Confusion Metrics",fontsize=20)

MSV_linear=sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')
MSV_linear.set_xlabel("Predicted label", fontsize = 10)
MSV_linear.set_ylabel("Actual label", fontsize = 10)

print(classification_report(y_test,poly_pred,target_names=levels))

# confusion metrics for SVM Radial Basis Function
rbf_pred = rbf.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

y,levels = pd.factorize(df['Label'])

#mat = confusion_matrix(y_test, y_predicted)
#rndf=sns.heatmap(mat,annot=True)
#rndf.set_xlabel("Predicted label", fontsize = 20)
#rndf.set_ylabel("Actual label", fontsize = 20)

cf_matrix = pd.crosstab(levels[y_test],levels[rbf_pred])
fig, ax = plt.subplots(figsize=(5,5),)
plt.title("MSV for RBF Model Confusion Matrics",fontsize=20)

MSV_linear=sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')
MSV_linear.set_xlabel("Predicted label", fontsize = 10)
MSV_linear.set_ylabel("Actual label", fontsize = 10)

print(classification_report(y_test,rbf_pred,target_names=levels))

# confusion metrics for Sigmoid
from sklearn.metrics import confusion_matrix,classification_report

sig_pred = sig.predict(X_test)
y,levels = pd.factorize(df['Label'])

#mat = confusion_matrix(y_test, y_predicted)
#rndf=sns.heatmap(mat,annot=True)
#rndf.set_xlabel("Predicted label", fontsize = 20)
#rndf.set_ylabel("Actual label", fontsize = 20)

cf_matrix = pd.crosstab(levels[y_test],levels[sig_pred])
fig, ax = plt.subplots(figsize=(5,5),)
plt.title("MSV for Sigmoid Model Confusion Matrics",fontsize=20)

MSV_linear=sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')
MSV_linear.set_xlabel("Predicted label", fontsize = 10)
MSV_linear.set_ylabel("Actual label", fontsize = 10)

print(classification_report(y_test,sig_pred,target_names=levels))



# retrieve the accuracy and print it for all 4 kernel functions
accuracy_lin = linear.score(X_test, y_test)
accuracy_poly = poly.score(X_test, y_test)
accuracy_rbf = rbf.score(X_test, y_test)
accuracy_sig = sig.score(X_test, y_test)
print("Accuracy Linear Kernel:", accuracy_lin)
print("Accuracy Polynomial Kernel:", accuracy_poly)
print("Accuracy Radial Basis Kernel:", accuracy_rbf)
print("Accuracy Sigmoid Kernel:", accuracy_sig)

cm_lin = confusion_matrix(y_test, linear_pred)
cm_poly = confusion_matrix(y_test, poly_pred)
cm_rbf = confusion_matrix(y_test, rbf_pred)
cm_sig = confusion_matrix(y_test, sig_pred)


 #please add the label in 
line=sns.heatmap(cm_lin,annot=True)
line.set_xlabel("Predicted label", fontsize = 20)
line.set_ylabel("Actual label", fontsize = 20)

poly=sns.heatmap(cm_poly,annot=True)
poly.set_xlabel("Predicted label", fontsize = 20)
poly.set_ylabel("Actual label", fontsize = 20)


radial=sns.heatmap(cm_rbf,annot=True)
radial.set_xlabel("Predicted label", fontsize = 20)
radial.set_ylabel("Actual label", fontsize = 20)


kernel=sns.heatmap(cm_sig,annot=True)
kernel.set_xlabel("Predicted label", fontsize = 20)
kernel.set_ylabel("Actual label", fontsize = 20)

# or print manually
#print(cm_lin)
#print(cm_poly)
#print(cm_rbf)
print(cm_sig)

#please include the f-score, accuracy etch
#how to save the model

#######################################################################

#classification using RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


df = pd.read_csv("C:/Users/PRO/Downloads/Untitled spreadsheet - Sheet1.csv")

print(df.describe().T)# To print the data information

print(df.isnull().sum()) #how many row do we have

#df = df.dropna() # to drop the null value

#Rename Dataset to Label to make it easy to understand
#df = df.rename(columns={'Diagnosis':'Label'}) # diagnosis from its orignal label to word label
#print(df.dtypes)

#Understand the data 
sns.countplot(x="Label", data=df) #apple,orange, banana

####### Replace categorical values with numbers########
print("Distribution of data: ", df['Label'].value_counts())

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values
print("Labels before encoding are: ", np.unique(y)) #it will print the value of the label

# Encoding categorical data from text (apple and orange and mango) to integers (0, 1 and 2)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y) # it will turn to numbercal value
print("Labels after encoding are: ", np.unique(Y))


#Define x and normalize / scale values

#Define the independent variables. Drop label and ID, and normalize other data
X_df = df.drop(labels = ["Label"], axis=1) # change the label and ID
print(X_df.describe().T) #Needs scaling

#Scale / normalize the values to bring them to similar range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_df)
X = scaler.transform(X_df)
print(X)  #Scaled values

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print("Shape of training data is: ", X_train.shape)




RFR_model = RandomForestClassifier(n_estimators = 40, random_state=30)
RFR_model.fit(X_train,y_train)

y_prediction = RFR_model.predict(X_test)


from sklearn.metrics import confusion_matrix,classification_report

y,levels = pd.factorize(df['Label'])

#mat = confusion_matrix(y_test, y_prediction)
#rndf=sns.heatmap(mat,annot=True)
#rndf.set_xlabel("Predicted label", fontsize = 20)
#rndf.set_ylabel("Actual label", fontsize = 20)

cf_matrix = pd.crosstab(levels[y_test],levels[ y_prediction])
fig, ax = plt.subplots(figsize=(5,5),)
plt.title("RandomForest Confusion Matrics")

RND=sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')
RND.set_xlabel("Predicted label", fontsize = 20)
RND.set_ylabel("Actual label", fontsize = 20)

#

from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,y_prediction,target_names=levels))

########################################################################

#classification using autokeras
"https://autokeras.com/structured_data_classifier/"
"https://autokeras.com/tutorial/overview/"


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("C:/Users/PRO/Downloads/Untitled spreadsheet - Sheet1.csv")

print(df.describe().T)# To print the data information

print(df.isnull().sum()) #how many row do we have

#df = df.dropna() # to drop the null value

#Rename Dataset to Label to make it easy to understand
#df = df.rename(columns={'Diagnosis':'Label'}) # diagnosis from its orignal label to word label
#print(df.dtypes)

#Understand the data 
sns.countplot(x="Label", data=df) #apple,orange, banana

####### Replace categorical values with numbers########
print("Distribution of data: ", df['Label'].value_counts())

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values
print("Labels before encoding are: ", np.unique(y)) #it will print the value of the label

# Encoding categorical data from text (B and M) to integers (0 and 1)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y) # it will turn to numbercal value
print("Labels after encoding are: ", np.unique(Y))

from  tensorflow.keras.utils import to_categorical 
Y=(to_categorical(Y))
print(Y)

#Define x and normalize / scale values

#Define the independent variables. Drop label and ID, and normalize other data
X_df = df.drop(labels = ["Label"], axis=1) # change the label and ID
print(X_df.describe().T) #Needs scaling

#Scale / normalize the values to bring them to similar range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_df)
X = scaler.transform(X_df)
print(X)  #Scaled values

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print("Shape of training data is: ", X_train.shape)
print("Shape of testing data is: ", X_test.shape)


# define the search
from autokeras import StructuredDataClassifier
...
# define the search
search = StructuredDataClassifier(max_trials=50)

# perform the search
search.fit(x=X_train, y=y_train, verbose=0, epochs=50)
loss, acc = search.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.3f' % acc)
print('loss: %.3f' % loss)


# get the best performing model
ak_model = search.export_model()

# summarize the loaded model
ak_model.summary()

#
from sklearn.metrics import confusion_matrix,classification_report

y_predicted = ak_model.predict(X_test)


mat = confusion_matrix(y_test.argmax(axis=1), y_predicted.argmax(axis=1))
 

y,levels = pd.factorize(df['Label'])

fig, ax = plt.subplots(figsize=(5,5),)
plt.title("AutoKeras model Confusion Matrics",fontsize=20)

auto_k=sns.heatmap(mat, linewidths=1, annot=True, ax=ax, fmt='g')
auto_k.set_xlabel("Predicted label", fontsize = 10)
auto_k.set_ylabel("Actual label", fontsize = 10)

print(classification_report(y_test.argmax(axis=1),y_predicted.argmax(axis=1),target_names=levels))


##########################################################
#Feature ranking using Random Forest

#Feature ranking...
import pandas as pd
feature_list = list(X.columns)
feature_imp = pd.Series(RFR_model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

from sklearn.model_selection import cross_val_score

scores_RFR = cross_val_score(RFR_model,X_train,y_train,scoring= 'r2', cv=10)
print(scores_RFR)

scores_SVM = cross_val_score(poly,X_train,y_train,scoring= 'r2', cv=10)
print(scores_SVM)
#k fold cross validation link
'https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md'

"10 fold cross validation"

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("C:/Users/PRO/Downloads/Untitled spreadsheet - Sheet1.csv")

print(df.describe().T)# To print the data information

print(df.isnull().sum()) #how many row do we have

#df = df.dropna() # to drop the null value

#Rename Dataset to Label to make it easy to understand
#df = df.rename(columns={'Diagnosis':'Label'}) # diagnosis from its orignal label to word label
#print(df.dtypes)

#Understand the data 
sns.countplot(x="Label", data=df) #apple,orange, banana

####### Replace categorical values with numbers########
print("Distribution of data: ", df['Label'].value_counts())

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values
print("Labels before encoding are: ", np.unique(y)) #it will print the value of the label

# Encoding categorical data from text (B and M) to integers (0 and 1)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y) # it will turn to numbercal value
print("Labels after encoding are: ", np.unique(Y))

from  tensorflow.keras.utils import to_categorical 
Y=(to_categorical(Y))
print(Y)

#Define x and normalize / scale values

#Define the independent variables. Drop label and ID, and normalize other data
X_df = df.drop(labels = ["Label"], axis=1) # change the label and ID
print(X_df.describe().T) #Needs scaling

#Scale / normalize the values to bring them to similar range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_df)
X = scaler.transform(X_df)
print(X)  #Scaled values

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print("Shape of training data is: ", X_train.shape)
print("Shape of testing data is: ", X_test.shape)


from sklearn.model_selection import KFold
import numpy as np
import statistics

num_folds = 10

# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []
f1=[]

# Merge inputs and targets
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# define the search
from autokeras import StructuredDataClassifier

fold_no = 1
for train, test in kfold.split(inputs, targets):
    # define the search
    search = StructuredDataClassifier(max_trials=100)

    # perform the search
    search.fit(x=inputs[train], y=targets[train], verbose=0, epochs=50)
    f1_score, acc= search.evaluate(inputs[test], targets[test], verbose=0)
    print('Accuracy: %.3f' % acc)
    print('loss: %.3f' % loss)
    acc_per_fold.append(acc)
    #loss_per_fold.append(loss)
    f1.append(f1_score)
print( acc_per_fold)
print(f1)
print(statistics.median(acc_per_fold))

    
