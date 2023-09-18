
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from featureExtraction import X_train, y_train


# Train the Bayesian Classifier
bayesian_classifier = MultinomialNB()
bayesian_classifier.fit(X_train, y_train)

# Train the Logistic Regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# Train the Support Vector Machines (SVM) model
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

