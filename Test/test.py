import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data.json"
DATA_PATH2="test.json"

def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["pitch"])
    y = np.array(data["labels"])
    z=np.array(data["mapping"])
    return X, y,z
# get train, validation, test splits
X, y,z = load_data(DATA_PATH)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0)

#Create a knn Classifier
model1 = KNeighborsClassifier(n_neighbors=5)
model2=AdaBoostClassifier(n_estimators=200,learning_rate=2)
model3=DecisionTreeClassifier()
model4=SVC(kernel='linear')
model5=SVC(kernel='rbf')

#Train the model using the training sets
model1.fit(X, y)
model2.fit(X, y)
model3.fit(X, y)
model4.fit(X, y)
model5.fit(X, y)

X_test, y_test, z_test = load_data(DATA_PATH2)
#Predict the response for test dataset
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5 = model5.predict(X_test)
#prediction of where_did:
print("KNN: ")
for i in range(len(X_test)):
    print(z[y_pred1[i]],end=' ' )
print("\nAdaboost: ")
for i in range(len(X_test)):
    print(z[y_pred2[i]],end=' ' )
print("\nDecision tree: ")
for i in range(len(X_test)):
    print(z[y_pred3[i]],end=' ' )
print("\nSVM rbf: ")
for i in range(len(X_test)):
    print(z[y_pred5[i]],end=' ' )
print("\nSVM linear: ")
for i in range(len(X_test)):
    print(z[y_pred4[i]],end=' ' )
print("\nReal chords: ")
print("em g em g em em em g em em g g em g g g em g g g em em g em em g g em em g g")
