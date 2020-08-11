import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



DATA_PATH = "data.json"


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
    return X, y
# get train, validation, test splits
X, y = load_data(DATA_PATH)
rounds =3
knn_accu=0
dt_accu=0
ada_accu=0
svm_accu=0

for i in range(rounds):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    #Create Classifiers:
    knn = KNeighborsClassifier(n_neighbors=5)
    svclassifier = SVC(kernel='linear')
    clf = DecisionTreeClassifier()
    abc = AdaBoostClassifier(n_estimators=200,
                            learning_rate=2)


    #Train the model using the training sets
    knn.fit(X_train, y_train)
    svclassifier.fit(X_train, y_train)
    clf = clf.fit(X_train,y_train)
    model = abc.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred1 = knn.predict(X_test)
    y_pred2 = svclassifier.predict(X_test)
    y_pred3 = clf.predict(X_test)
    y_pred4 = model.predict(X_test)

    knn_accu+=metrics.accuracy_score(y_test, y_pred1)
    dt_accu+=metrics.accuracy_score(y_test, y_pred2)
    ada_accu+=metrics.accuracy_score(y_test, y_pred3)
    svm_accu+=metrics.accuracy_score(y_test, y_pred4)
    print(i)


print()
print("knn avarage accuracy: " , knn_accu/rounds)
print("Decision Tree avarage accuracy: " , dt_accu/rounds)
print("Adaboost avarage accuracy: " , ada_accu/rounds)
print("SVM avarage accuracy: " , svm_accu/rounds)