import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Create a Decision tree Classifier
clf = DecisionTreeClassifier()

#Train the model using the training sets
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


print("test size: ", len(y_test))
print("train size: ",len(y_train))
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

err_count=0
for i in range(len(y_test)):
    if y_test[i]!=y_pred[i]:
        err_count+=1
        print("error in:",i," test: ",y_test[i], "pred: ", y_pred[i])

print("number of errors: ",err_count, "in ",len(y_test),"in total:",err_count/len(y_test) )
