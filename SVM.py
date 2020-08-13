import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC


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

#Create a SVM Classifier
svclassifier_lin = SVC(kernel='linear')
svclassifier_rbf = SVC(kernel='rbf')

#Train the model using the training sets
svclassifier_lin.fit(X_train, y_train)
svclassifier_rbf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_lin = svclassifier_lin.predict(X_test)
y_pred_rbf = svclassifier_rbf.predict(X_test)


print("test size: ", len(y_test))
print("train size: ",len(y_train))
# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

err_count_lin=0
err_count_rbf=0
for i in range(len(y_test)):
    if y_test[i]!=y_pred_rbf[i]:
        err_count_rbf+=1
        #print("error in:",i," test: ",y_test[i], "pred: ", y_pred[i])
    if y_test[i]!=y_pred_lin[i]:
        err_count_lin+=1

print("~~~Errors Comparison: ~~~")
print("Type of SVM: Linear , number of errors: ", err_count_lin, "accuracy: ", metrics.accuracy_score(y_test, y_pred_lin))
print("Type of SVM: radial basis function , number of errors: ", err_count_rbf, "accuracy: ", metrics.accuracy_score(y_test, y_pred_rbf))