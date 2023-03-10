# Classification-of-Fruit-using-Artificial-Neural-Network

The code imports necessary libraries and reads a CSV file containing data on fruit attributes. The data is then explored using describe and isnull functions. The distribution of data is visualized using a countplot.

The label column is encoded using LabelEncoder and then converted to one-hot encoding using to_categorical. The features data is then normalized using MinMaxScaler.

The normalized data is then split into training and testing data using train_test_split and used to create an artificial neural network model using Keras. The model is then trained using the training data and validated using the testing data. The accuracy and loss of the model are then visualized using matplotlib.

The model's accuracy is further evaluated using a confusion matrix using confusion_matrix and visualized using heatmap from seaborn. The code also includes an example of using SVM for classification, which is not implemented.
