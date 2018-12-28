# kaggle_titanic

### About
This repository contains the code that achieves 76% score in the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/overview)
The code was developed to serve as an initial exploration of Scikit-Learn.

### Pre-Processing 
To achieve the result, the following were the pre-processing techniques applied:

- [Age], [Fare]: Missing Values were handled with Mean
- [Embarked]: Missing values were handled with Most Frequent Strategy
- [Embarked]: One Hot Encoder applied
- [Sex]: Ordinal Encoder applied
- [Age], [SibSp], [Fare], [Pclass]: Numerical Features that were normalized with MinMaxScaler

### Model
The model built in this implementation is a simple Logistic Regression Model

