
import pandas as pd #Imports necessary libraries: pandas for data manipulation
import numpy as np #numpy for numerical operations,
import pickle #pickle for saving the model

df = pd.read_csv('iris.data') #Reads the dataset: Loads the Iris dataset from a CSV file into a DataFrame df

#Separates features and labels:
X = np.array(df.iloc[:, 0:4]) #X contains the feature columns (first four columns of the dataset).           
y = np.array(df.iloc[:, 4:])  # y contains the label column (fifth column of the dataset).

# Encodes the labels:
#Imports LabelEncoder from sklearn.preprocessing.
#Initializes the encoder le.
#Transforms the labels y into a numeric format using fit_transform
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y.reshape(-1))

# Splits the dataset:
#Imports train_test_split from sklearn.model_selection.
#Splits the data into training and testing sets with 80% for training and 20% for testing.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Trains the model:
#Imports SVC (Support Vector Classifier) from sklearn.svm.
#Initializes the classifier with a linear kernel.
#Fits the model on the training data (X_train, y_train).
from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train,y_train)

# Saves the model:
#Uses pickle.dump to serialize the trained model sv.
#Saves the serialized model to a file named iri.pkl.
pickle.dump(sv, open('iri.pkl', 'wb'))
