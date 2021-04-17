import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split 


from decision_tree import DecisionTree
from random_forest import RandomForest

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy


data = datasets.load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# # Decision Tree
# clf = DecisionTree(max_depth=10)    # Instantiate
# clf.fit(X_train, y_train)           # Fit Model     
# y_pred = clf.predict(X_test)        # Predict
# dt_accuracy = accuracy(y_test, y_pred)
# print(dt_accuracy)

# from sklearn.tree import DecisionTreeClassifier
# dtree = DecisionTree()          # Instantiate
# dtree.fit(X_train, y_train)     # Fit Model  
# y_pred = dtree.predict(X_test)  # Predict
# dt_accuracy = accuracy(y_test, y_pred)
# print(dt_accuracy)


# # Random Forest
clf = RandomForest(n_trees=3)    # Instantiate
clf.fit(X_train, y_train)           # Fit Model     
y_pred = clf.predict(X_test)        # Predict
rf_accuracy = accuracy(y_test, y_pred)
print(rf_accuracy)


