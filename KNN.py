import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

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

#Create a knn Classifier
knn1 = KNeighborsClassifier(n_neighbors=1)
knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn7 = KNeighborsClassifier(n_neighbors=7)

#Train the model using the training sets
knn1.fit(X_train, y_train)
knn3.fit(X_train, y_train)
knn5.fit(X_train, y_train)
knn7.fit(X_train, y_train)

#Predict the response for test dataset
y_pred1 = knn1.predict(X_test)
y_pred3 = knn3.predict(X_test)
y_pred5 = knn5.predict(X_test)
y_pred7 = knn7.predict(X_test)


print("test size: ", len(y_test))
print("train size: ",len(y_train))

# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred3))
err_count1=0
err_count3=0
err_count5=0
err_count7=0

for i in range(len(y_test)):
    if y_test[i]!=y_pred1[i]:
        err_count1+=1
    if y_test[i]!=y_pred3[i]:
        err_count3+=1
    if y_test[i]!=y_pred5[i]:
        err_count5+=1
    if y_test[i]!=y_pred7[i]:
        err_count7+=1
print("~~~Errors Comparison: ~~~")
print("Number of neighbor= ",1," ,number of errors: ",err_count1 ,", accuracy: ",metrics.accuracy_score(y_test, y_pred1) )
print("Number of neighbor= ",3," ,number of errors: ",err_count3 ,", accuracy: ",metrics.accuracy_score(y_test, y_pred3) )
print("Number of neighbor= ",5," ,number of errors: ",err_count5,", accuracy: ",metrics.accuracy_score(y_test, y_pred5) )
print("Number of neighbor= ",7," ,number of errors: ",err_count7,", accuracy: ",metrics.accuracy_score(y_test, y_pred7) )
