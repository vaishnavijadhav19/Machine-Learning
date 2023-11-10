import math
import numpy as np 
import pandas as pd 
import seaborn as sns 
from seaborn import countplot
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure,show 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

def Titanic():

    data = pd.read_csv('TitanicDataset.csv')

    print("First 5 entries from the data set")
    print(data.head())

    print("Number of passengers are:"+str(len(data)))

    print("Visualization: Survived and Non survived passengers")
    figure()
    target = "Survived"
    countplot(data = data, x = target).set_title(" Survived and not survived passengers")
    show()

    print("Visualization: Survived and Non survived passengers based on gender")
    figure()
    target = "Survived"
    countplot(data = data, x = target, hue = "Sex").set_title(" Survived and not survived passengers based on genders")
    show()

    print("Visualization: Survived and Non survived passengers based on passenger class")
    figure()
    target = "Survived"
    countplot(data = data, x = target, hue = "Pclass").set_title(" Survived and not survived passengers based on passenger class")
    show()

    print("Visualization: Survived and Non survived passengers based on age")
    figure()
    target = "Survived"
    data["Age"].plot.hist().set_title("Survived and Non survived passengers based on Age")
    show()

    print("Visualization: Survived and Non survived passengers based on Fare")
    figure()
    target = "Survived"
    data["Fare"].plot.hist().set_title("Survived and Non survived passengers based on fare")
    show()

    data.drop("zero", axis = 1, inplace = True)

    print("First five entries from the loaded dataset after removing zero")
    print(data.head())

    print("Values of sex column")
    print(pd.get_dummies(data["Sex"]))

    print("Values of sex column after removing one field")
    Sex = pd.get_dummies(data["Sex"], drop_first = True)
    print(Sex.head(5))

    print("Values of pclass column after removing one field")
    Pclass = pd.get_dummies(data["Pclass"], drop_first = True)
    print(Pclass.head(5))

    print("Values of dataset after concatenating new columns")
    data = pd.concat([data,Sex,Pclass], axis = 1)
    print(data.head(5))

    print("Values of dataset after removing irrelavant columns")
    data.drop(["Sex", "sibsp", "Parch", "Embarked"], axis = 1, inplace = True)
    print(data.head(5))

    x = data.drop("Survived", axis = 1)
    y = data["Survived"]

    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.5)

    xtrain.columns = xtrain.columns.astype(str)
    xtest.columns = xtest.columns.astype(str)


    model = LogisticRegression(max_iter=1000)  


    model.fit(xtrain, ytrain)

    prediction = model.predict(xtest)

    print("Classification report of logistic regression is :")
    print(classification_report(ytest, prediction))

    print("Confusion matrix of logistic regression is :")
    print(confusion_matrix(ytest, prediction))

    print("Accuracy of logistic regression is :")
    print(accuracy_score(ytest,prediction))



def main():

    print("Name of the Case study: Titanic Survival")
    print("Algorithm used: Linear Regression")

    Titanic()

if __name__ == "__main__":
    main()

