import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np
from sklearn import cross_validation

titanic_train_data = pd.read_csv("../data/train.csv")

#Replacing not available age data with the median
titanic_train_data["Age"] = titanic_train_data["Age"].fillna(titanic_train_data["Age"].median())

#Replacing all the values in sex column with numeric data. Male = 0 and Female =1
titanic_train_data.loc[titanic_train_data["Sex"] == "male", "Sex"] = 0
titanic_train_data.loc[titanic_train_data["Sex"] == "female", "Sex"] = 1

#Replacing all the values in Embarked column with numeric data

titanic_train_data["Embarked"] = titanic_train_data["Embarked"].fillna("S")
titanic_train_data.loc[titanic_train_data["Embarked"] == "S", "Embarked"] = 0
titanic_train_data.loc[titanic_train_data["Embarked"] == "C", "Embarked"] = 1
titanic_train_data.loc[titanic_train_data["Embarked"] == "Q", "Embarked"] = 2

#Now Let's use linear regression to predict the survivor status.
# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic_train_data.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic_train_data[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic_train_data["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic_train_data[predictors].iloc[test,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

accuracy = sum(predictions[predictions == titanic_train_data["Survived"]]) / len(predictions)
print "Accuracy With Linear Regression" +str(accuracy)

# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic_train_data[predictors], titanic_train_data["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print "Accuracy With Logistic Regression" +str(scores.mean())