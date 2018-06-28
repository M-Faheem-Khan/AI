# Muhammad Faheem Khan
# This program is a simple challage code from https://www.youtube.com/watch?v=T5pRlIbr6gg&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU
# Challage: Use 3 different models to predict wheather some one is a male or female based on height, weight, shoe size
# algorithms used: Decision Tree, Support Vector Machine(Support Vecotr Classifier), Preceptron, KNN - K nearest neighbor

# imports
from sklearn import tree # Decision Tree
from sklearn import svm # Support Vector Machine(support vector classification (svc))
from sklearn.metrics import accuracy_score # Get the prediction accuracy
from sklearn.linear_model import Perceptron # Preceptron model
from sklearn.neighbors import KNeighborsClassifier
import os # operating system for clearing console

# Import the model
# Create classifier object
# train the data (fit())
# predict(test)

# Labels
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
    [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# Genders
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# classifers
# Decision Tree classifer
clf_tree = tree.DecisionTreeClassifier()
# Support Vector classification
clf_svc = svm.SVC()
# Perceptron classifer
clf_precpt = Perceptron()
# K-nearest Neighbours
clf_knn = KNeighborsClassifier()


# Training (fit)
clf_tree_fit = clf_tree.fit(X, Y)
clf_svc_fit = clf_svc.fit(X, Y)
clf_precpt_fit = clf_precpt.fit(X, Y)
clf_knn_fit = clf_knn.fit(X, Y)

# Predicting
clf_tree_predicta = clf_tree.predict(X)
clf_svc_predicta = clf_svc.predict(X)
clf_precpt_predicta = clf_precpt.predict(X)
clf_knn_predicta = clf_knn.predict(X)

os.system("cls")

# Printing accuracy score
# Decision tress
acc_decsision_tree = accuracy_score(Y, clf_tree_predicta) * 100
print('Accuracy for Decision Treee: {} %'.format(round(acc_decsision_tree), 5))

# SVC classifer
acc_svc = accuracy_score(Y, clf_svc_predicta) * 100
print('Accuracy for Support Vecotr classifier: {} %'.format(round(acc_svc), 5))

# Preceptron classifier
acc_precpt = accuracy_score(Y, clf_precpt_predicta) * 100
print('Accuracy for Preceptron Classifier: {} %'.format(round(acc_precpt), 5))

# K-nearest neighbors
acc_knn = accuracy_score(Y, clf_knn_predicta) * 100
print('Accuracy for K-nearest Neightbour: {} %'.format(round(acc_knn), 5))
