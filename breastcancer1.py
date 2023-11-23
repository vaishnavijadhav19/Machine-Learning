
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



inputpath = "BC.data"
outputpath = "BC.csv"

Headers = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion", "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli","Mitoses","CancerType"]


def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features,target)
    return clf

def split_dataset(dataset, train_percentage,feature_headers,target_header):
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers],dataset[target_header],train_size=train_percentage)

    return train_x, test_x, train_y, test_y



def handel_missing_values(dataset, missing_values_header,missing_label):
    return dataset[dataset[missing_values_header] != missing_label]



def dataset_statistics(dataset):
    print(dataset.describe())


def main():
    dataset = pd.read_csv(outputpath)

    #Get basic statistics

    dataset_statistics(dataset)

    #filter missing values
    dataset = handel_missing_values(dataset, Headers[6],'?')
    train_x, test_x, train_y, test_y = split_dataset(dataset,0.7,Headers[1:-1],Headers[-1])


    print("Train_x Shape ::",train_x.shape)
    print("Train_y Shape ::",train_y.shape)
    print("Test_x Shape ::",test_x.shape)
    print("Test_y Shape ::",test_y.shape)



    trainedmodel = random_forest_classifier(train_x,train_y)
    print("Trained Model ::",trainedmodel)
    predictions = trainedmodel.predict(test_x)

    for i in range(0,205):
        print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i],predictions[i]))


    print("Train Accuracy :: ", accuracy_score(train_y, trainedmodel.predict(train_x)))
    print("Test Accuracy :: ",accuracy_score(test_y, predictions))
    print("Confusion Matrix ",confusion_matrix(test_y, predictions))







    


if __name__ == "__main__":
    main()