# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:47:44 2016

@author: Manuel
"""

from C45Tree_own import tree
from C45Tree_own import tree_apply
import pandas as pa
import sklearn.metrics as sklm
import sklearn.tree as sklt


#import
train_data = pa.read_csv("Data/admit-train.csv", sep = ",")
test_data = pa.read_csv("Data/admit-test.csv", sep = ",")

#fitting several models of my implementation
model1 = tree.fit(train_data.loc[:,"admit":"gpa"], train_data.loc[:,"rank"])
print("model1 is fitted")
model2 = tree.fit(train_data.loc[:,"admit":"gpa"], train_data.loc[:,"rank"], branching = "giniIndex", splitCriterion = "giniIndex")
print("model2 is fitted")

#fitting a model with sklearn (gini), equal to model2

tree3 = sklt.DecisionTreeClassifier(max_features = "auto")
model3 = tree3.fit(X = train_data.loc[:,"admit":"gpa"].values, y = train_data.loc[:,"rank"].values)
print("model3 is fitted")

tree4 = sklt.DecisionTreeClassifier()
model4 = tree4.fit(X = train_data.loc[:,"admit":"gpa"].values, y = train_data.loc[:,"rank"].values)
print("model4 is fitted")

##Build predictions on test data
test_data = pa.read_csv("Data/admit-test.csv", sep = ",")

estimations_model1 = tree_apply.apply(test_data.loc[:,"admit":"gpa"], model1)
estimations_model2 = tree_apply.apply(test_data.loc[:,"admit":"gpa"], model2)
estimations_model3 = model3.predict(test_data.loc[:,"admit":"gpa"].values)
estimations_model4 = model4.predict(test_data.loc[:,"admit":"gpa"].values)

#compare the metrics
print("#### EVALUATION MODEL 1 ####")
print()
print("accuracy")
print(sklm.accuracy_score(test_data.loc[:,"rank"].values, estimations_model1))
print()
print("confusion matrix")
print(pa.DataFrame(sklm.confusion_matrix(test_data.loc[:,"rank"].values, estimations_model1), index = [1,2,3,4], columns = [1,2,3,4]))

print("#### EVALUATION MODEL 2 ####")
print()
print("accuracy")
print(sklm.accuracy_score(test_data.loc[:,"rank"].values, estimations_model2))
print()
print("confusion matrix")
print(pa.DataFrame(sklm.confusion_matrix(test_data.loc[:,"rank"].values, estimations_model2), index = [1,2,3,4], columns = [1,2,3,4]))

print("#### EVALUATION MODEL 3 ####")
print()
print("accuracy")
print(sklm.accuracy_score(test_data.loc[:,"rank"].values, estimations_model3))
print()
print("confusion matrix")
#print(pa.DataFrame(sklm.confusion_matrix(test_data.loc[:,"rank"].values, estimations_model3)))

print("#### EVALUATION OF M1 ON TRAIN DATA ####")
print()
print("accuracy")
print(sklm.accuracy_score(train_data.loc[:,"rank"].head(len(test_data.loc[:,"rank"].values)).values, estimations_model1))
print()
print("confusion matrix")
print(pa.DataFrame(sklm.confusion_matrix(train_data.loc[:,"rank"].head(len(test_data.loc[:,"rank"].values)).values, estimations_model1), index = [1,2,3,4], columns = [1,2,3,4]))
