# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


def performPreprocessing(titanic):

    # dropping a few features
    titanic = titanic.drop(['Name','Cabin','Ticket'], axis=1)

    # checking missing values
    #print(titanic.isnull().sum())

    # handling missing values
    # applying mean for Age and Fare features
    # applying most_frequent for Embarked
    imputer_mean = SimpleImputer(missing_values=np.NaN, strategy='mean')
    imputer_frequent = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')

    imputer_mean.fit(titanic[['Age']])
    titanic['Age'] = imputer_mean.transform(titanic[['Age']])

    imputer_mean.fit(titanic[['Fare']])
    titanic['Fare'] = imputer_mean.transform(titanic[['Fare']])

    imputer_frequent.fit(titanic[['Embarked']])
    titanic['Embarked'] = imputer_frequent.transform(titanic[['Embarked']])

    # categorical features handling
    # embarked: ordinal Feature
    # Sex: nominal feature

    # handling with mapping
    # embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    # titanic['Embarked'] = titanic['Embarked'].map(embarked_mapping)

    # handling Embarked with OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    embarkEncoded = encoder.fit_transform( titanic[["Embarked"]] )
    embarkedDF = pd.DataFrame(embarkEncoded)
    #print(embarkedDF)
    
    # # concatenating Embarked OneHotEncoder
    frameList = [embarkedDF, titanic]
    titanic = pd.concat(frameList, axis=1)      
    # dropping old Embarked
    titanic = titanic.drop(['Embarked'], axis=1)    

    # sex encoder    
    encoder = OrdinalEncoder()
    titanic['Sex'] = encoder.fit_transform(titanic[['Sex']])

    # normalizing data
    # features Age, SibSp, Fare, Pclass
    scalingObj= preprocessing.MinMaxScaler()
    titanic[['Age', 'SibSp', 'Fare', 'Pclass']]= scalingObj.fit_transform( titanic[['Age', 'SibSp', 'Fare', 'Pclass']] )
  

    return titanic

def break_train_test(titanic_data):

    # breaking down the data into Train and Test
    mask = titanic_data['Survived'] >= 0    
    train = titanic_data[mask]
    # ~ is the inverted portion
    test = titanic_data[~mask]     

    return [train, test]

    
def model(titanic_train, titanic_test):

   # splitting training into data and prediction. required to build the model
    titanic_train_target = titanic_train.iloc[:,-1]
    titanic_train_data = titanic_train.loc[:, titanic_train.columns != 'Survived']

    titanic_test_target = titanic_test.iloc[:,-1]
    titanic_test_data =  titanic_test.loc[:, titanic_test.columns != 'Survived']   

    # dropping PassengerID from both training and test dataset. But storying the test as we will need it later to submit to Kaggle        
    # dropping for train first
    titanic_train_data = titanic_train_data.drop(['PassengerId'], axis=1)
    
    # dropping for test but keeping it separate
    titanic_test_PassengerId = titanic_test['PassengerId']

    titanic_test_data = titanic_test_data.drop(['PassengerId'], axis=1)

    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    clf.fit(titanic_train_data,titanic_train_target) 
    results = clf.predict(titanic_test_data)

    # results below is the NumPy array of predicted results returned from our classifier
    resultSeries = pd.Series(data = results, name = 'Survived', dtype='int64')

    # important to reset index of the passengetID dataframe so that the data for result and passengerId is correctly concatenated
    titanic_test_PassengerId = titanic_test_PassengerId.reset_index(drop=True)

    # create a data frame with just the PassengerID feature from the test dataset and the results
    df = pd.DataFrame({"PassengerId":titanic_test_PassengerId, "Survived":resultSeries})
    # write the results to a CSV file (you should then upload this file)
    df.to_csv("submission.csv", index=False, header=True)      

def load_dataset():
    # Open the training and test dataset as a pandas dataframe
    train = pd.read_csv("train.csv", delimiter=",")
    test = pd.read_csv("test.csv", delimiter=",")

    # Merge the two datasets into one dataframe so we can perform preprocessing on all data at once    
    test["Survived"] = np.zeros(len(test))
    test["Survived"] = -1
    frameList = [train, test]
    allData = pd.concat(frameList, ignore_index=True)

    return allData

def main():
    
    allData = load_dataset()
        
    # run preprocessing.
    all_data = performPreprocessing(allData)

    # seperate the resulting data into test and train
    train, test = break_train_test(all_data)

    model(train, test)

  
main()
