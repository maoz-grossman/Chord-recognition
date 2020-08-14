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

rounds= 1000
Acc_knn1_avg=0
Acc_knn3_avg=0
Acc_knn5_avg=0
Acc_knn7_avg=0

err_count1=0
err_count3=0
err_count5=0
err_count7=0

#LOOP:  
for i in range(rounds):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    #Create a SVM Classifier
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

    # Model Accuracy, how often is the classifier correct?
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    Acc_knn1_avg+=metrics.accuracy_score(y_test, y_pred1)
    Acc_knn3_avg+=metrics.accuracy_score(y_test, y_pred3)
    Acc_knn5_avg+=metrics.accuracy_score(y_test, y_pred5)
    Acc_knn7_avg+=metrics.accuracy_score(y_test, y_pred7)


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
print("Number of neighbor= ",1," ,number of errors: ",err_count1/rounds,", accuracy: ",Acc_knn1_avg/rounds )
print("Number of neighbor= ",3," ,number of errors: ",err_count3/rounds,", accuracy: ",Acc_knn3_avg/rounds )
print("Number of neighbor= ",5," ,number of errors: ",err_count5/rounds,", accuracy: ",Acc_knn5_avg/rounds )
print("Number of neighbor= ",7," ,number of errors: ",err_count7/rounds,", accuracy: ",Acc_knn7_avg/rounds )