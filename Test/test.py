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
model_knn = KNeighborsClassifier(n_neighbors=3)
model_ada=AdaBoostClassifier(n_estimators=200,learning_rate=2)
model_dt=DecisionTreeClassifier()
model_svc_lin=SVC(kernel='linear')
model_svc_rbf=SVC(kernel='rbf')

#Train the model using the training sets
model_knn.fit(X, y)
model_ada.fit(X, y)
model_dt.fit(X, y)
model_svc_lin.fit(X, y)
model_svc_rbf.fit(X, y)

X_test, y_test, z_test = load_data(DATA_PATH2)
#Predict the response for test dataset
y_pred_knn = model_knn.predict(X_test)
y_pred_ada = model_ada.predict(X_test)
y_pred_dt = model_dt.predict(X_test)
y_pred_svm_lin = model_svc_lin.predict(X_test)
y_pred_svm_rbf = model_svc_rbf.predict(X_test)
#prediction of where_did:
print("KNN: ")
for i in range(len(X_test)):
    print(z[y_pred_knn[i]],end=' ' )
print("\nAdaboost: ")
for i in range(len(X_test)):
    print(z[y_pred_ada[i]],end=' ' )
print("\nDecision tree: ")
for i in range(len(X_test)):
    print(z[y_pred_dt[i]],end=' ' )
print("\nSVM rbf: ")
for i in range(len(X_test)):
    print(z[y_pred_svm_rbf[i]],end=' ' )
print("\nSVM linear: ")
for i in range(len(X_test)):
    print(z[y_pred_svm_lin[i]],end=' ' )
print("\nReal chords: ")
print("em g em g em em em g em em g g em g g g em g g g em em g em em g g em em g g")
