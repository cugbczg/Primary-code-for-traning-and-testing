import pandas as pd
import numpy as np
# Import training set data file 

train=pd.read_csv("D:/2018_BigData/Python/Kaggle_learning/Titanic Machine Learning from Disaster/titanic/train2.csv")

train.head
# Import test set data file
test=pd.read_csv("D:/2018_BigData/Python/Kaggle_learning/Titanic Machine Learning from Disaster/titanic/test2.csv")
test.head(5)

# Try the random forest algorithm, first throw in the variables that are somewhat related

from sklearn.ensemble import RandomForestClassifier
x1 = train[["Pclass","Fare","Family","age_group0","Sex0","Embarked0"]]
y1 = train["Survived"]
x_test1 = test[["Pclass","Fare","Family","age_group0","Sex0","Embarked0"]]

random_forest = RandomForestClassifier(oob_score=True, n_estimators=1000)
random_forest.fit(x1, y1)

Y_pred = random_forest.predict(x_test1)
score_randomforest = random_forest.score(x1, y1)
score_randomforest



Final = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred.astype(int)})
Final.to_csv(r"D:/2018_BigData/Python/Kaggle_learning/Titanic Machine Learning from Disaster/Final7-randomforest.csv",index=False)





x1 = train[["Pclass","Fare","Family","age_group0","Sex0","Embarked0"]]
y1 = train["Survived"]
x_test1 = test[["Pclass","Fare","Family","age_group0","Sex0","Embarked0"]]

random_forest = RandomForestClassifier(oob_score=True, n_estimators=500)
random_forest.fit(x1, y1)

Y_pred = random_forest.predict(x_test1)
score_randomforest = random_forest.score(x1, y1)
score_randomforest

Final = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred.astype(int)})
Final.to_csv(r"D:/2018_BigData/Python/Kaggle_learning/Titanic Machine Learning from Disaster/Final9-randomforest.csv",index=False)

# kaggle score 0.77990


x1 = train[["Pclass","Fare","Family","age_group0","Sex0","Embarked0"]]
y1 = train["Survived"]
x_test1 = test[["Pclass","Fare","Family","age_group0","Sex0","Embarked0"]]

random_forest = RandomForestClassifier(oob_score=True, n_estimators=1200)
random_forest.fit(x1, y1)

Y_pred = random_forest.predict(x_test1)
score_randomforest = random_forest.score(x1, y1)
score_randomforest

Final = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred.astype(int)})
Final.to_csv(r"D:/2018_BigData/Python/Kaggle_learning/Titanic Machine Learning from Disaster/Final10-randomforest.csv",index=False)


x2 = train[["Pclass","Family","Sex0"]]
y2 = train["Survived"]
x_test2 = test[["Pclass","Family","Sex0"]]

random_forest = RandomForestClassifier(oob_score=True, n_estimators=1000)
random_forest.fit(x2, y2)

Y_pred = random_forest.predict(x_test2)
score_randomforest = random_forest.score(x2, y2)
score_randomforest



Final = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred.astype(int)})
Final.to_csv(r"D:/2018_BigData/Python/Kaggle_learning/Titanic Machine Learning from Disaster/Final8-randomforest.csv",index=False)

# kaggle score 0.77033


x3 = train[["Pclass","Family","age_group0","Sex0","Embarked0"]]
y3 = train["Survived"]
x_test3 = test[["Pclass","Family","age_group0","Sex0","Embarked0"]]

random_forest = RandomForestClassifier(oob_score=True, n_estimators=1000)
random_forest.fit(x3, y3)

Y_pred = random_forest.predict(x_test3)
score_randomforest = random_forest.score(x3, y3)
score_randomforest

Final = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred.astype(int)})
Final.to_csv(r"D:/2018_BigData/Python/Kaggle_learning/Titanic Machine Learning from Disaster/Final11-randomforest.csv",index=False)



x4 = train[["Fare","Family","age_group0","Sex0","Embarked0"]]
y4 = train["Survived"]
x_test4 = test[["Fare","Family","age_group0","Sex0","Embarked0"]]

random_forest = RandomForestClassifier(oob_score=True, n_estimators=1000)
random_forest.fit(x4, y4)

Y_pred = random_forest.predict(x_test4)
score_randomforest = random_forest.score(x4, y4)
score_randomforest

Final = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred.astype(int)})
Final.to_csv(r"D:/2018_BigData/Python/Kaggle_learning/Titanic Machine Learning from Disaster/Final12-randomforest.csv",index=False)


